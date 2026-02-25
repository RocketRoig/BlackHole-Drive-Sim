import datetime as dt
import math
import random
from dataclasses import dataclass
import pygame

WIDTH, HEIGHT = 1280, 800
CENTER = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
G = 6.67430e-11
C = 299_792_458
G0 = 9.80665
AU = 1.496e11
TIME_STEPS = [1.0, 10.0, 100.0, 10_000.0, 1_000_000.0]
MAX_SHIP_TRAIL_POINTS = 2400
ORBIT_SHAPE_SAMPLES = 220
HAWKING_MASS_LOSS_MAX = 1_000_000.0  # kg/s saturation near end-of-life
BH_MIN_MASS = 1_000.0  # kg, below this BH is considered evaporated
SHOCKWAVE_VISUAL_SCALE = 30.0  # visual amplification so expansion is visible on screen
HARVEST_INFLUENCE_RADIUS = 0.1 * AU
HARVEST_MAX_FLOW = 1000.0  # kg/s (1 ton/s)

# Reference black-hole drive model (from design note)
M0 = 1.066e8  # kg (nominal black-hole mass)
MDOT0 = 0.352  # kg/s (nominal Hawking mass-loss when at M0)
P0 = MDOT0 * C**2  # W (nominal Hawking power)
BH_LIFETIME_AT_M0 = 1.0175e8  # s (if feeding stops at M0)
BH_LIFETIME_K = BH_LIFETIME_AT_M0 / (M0**3)  # t(M)=k*M^3


@dataclass
class StellarBody:
    name: str
    pos: pygame.Vector2
    mass: float
    radius: float
    color: tuple[int, int, int]
    harvest_rate: float
    orbit_radius: float
    orbit_period_days: float
    phase_offset: float
    is_planet: bool = True


class BlackHoleDrive:
    def __init__(self):
        self.mass = M0
        self.feed_massflow = 0.0
        self.efficiency = 0.99
        self.nominal_limit = P0
        self.failed = False
        self.exists = True

    @property
    def irradiation(self) -> float:
        if not self.exists:
            return 0.0
        return P0 * (M0 / self.mass) ** 2

    @property
    def hawking_mass_loss_rate(self) -> float:
        return min(HAWKING_MASS_LOSS_MAX, self.irradiation / (C**2))

    @property
    def thrust(self) -> float:
        if not self.exists:
            return 0.0
        return self.efficiency * self.irradiation / C

    @property
    def irradiation_ratio(self) -> float:
        return self.irradiation / self.nominal_limit if self.nominal_limit > 0 else 0.0

    @property
    def time_to_2x_power_if_unfed(self) -> float:
        # 2x power occurs at M = M0 / sqrt(2)
        target_mass = M0 / math.sqrt(2)
        if self.mass <= target_mass:
            return 0.0
        return BH_LIFETIME_K * (self.mass**3 - target_mass**3)

    def update(self, dt_seconds: float, available_fuel: float) -> tuple[float, float]:
        if not self.exists:
            return 0.0, 0.0
        feed = min(available_fuel / dt_seconds if dt_seconds > 0 else 0.0, self.feed_massflow)
        self.mass += feed * dt_seconds
        evaporated = self.hawking_mass_loss_rate * dt_seconds
        self.mass = self.mass - evaporated
        if self.mass <= BH_MIN_MASS:
            self.mass = 0.0
            self.exists = False
            return feed * dt_seconds, evaporated
        return feed * dt_seconds, evaporated


class Ship:
    def __init__(self):
        self.pos = pygame.Vector2(1.05 * AU, 0)
        self.vel = pygame.Vector2(0, 31_500)
        self.heading = math.pi / 2
        self.mass = 1e7
        self.cargo_mass = 5e7
        self.max_cargo = 10e9
        self.aligned_for_thrust = False
        self.destroyed = False
        self.blackhole = BlackHoleDrive()

    def total_mass(self) -> float:
        return self.mass + self.cargo_mass + self.blackhole.mass


def mean_anomaly_phase(orbit_period_days: float) -> float:
    epoch = dt.datetime(2000, 1, 1, 12, 0, 0)
    now = dt.datetime.utcnow()
    elapsed_days = (now - epoch).total_seconds() / 86400.0
    return 2 * math.pi * ((elapsed_days / orbit_period_days) % 1.0)


def gravity_accel(pos: pygame.Vector2, bodies: list[StellarBody]) -> pygame.Vector2:
    acc = pygame.Vector2(0, 0)
    for body in bodies:
        delta = body.pos - pos
        if delta.length_squared() == 0:
            continue
        r2 = max(delta.length_squared(), (body.radius * 1.2) ** 2)
        acc += delta.normalize() * (G * body.mass / r2)
    return acc


def compute_orbit_geometry(position: pygame.Vector2, velocity: pygame.Vector2, mu: float):
    r = position.length()
    v2 = velocity.length_squared()
    if r <= 0:
        return None

    energy = 0.5 * v2 - mu / r
    if energy >= 0:
        return None

    h = position.x * velocity.y - position.y * velocity.x
    if abs(h) < 1e-9:
        return None

    e_vec = pygame.Vector2((velocity.y * h) / mu - position.x / r, (-velocity.x * h) / mu - position.y / r)
    e = e_vec.length()
    if e >= 1:
        return None

    a = -mu / (2 * energy)
    b = a * math.sqrt(max(0.0, 1 - e * e))
    peri_angle = math.atan2(e_vec.y, e_vec.x) if e > 1e-10 else math.atan2(position.y, position.x)
    center = pygame.Vector2(math.cos(peri_angle), math.sin(peri_angle)) * (-a * e)
    apogee = pygame.Vector2(math.cos(peri_angle), math.sin(peri_angle)) * (-a * (1 + e))

    points: list[pygame.Vector2] = []
    for i in range(ORBIT_SHAPE_SAMPLES):
        t = 2 * math.pi * i / ORBIT_SHAPE_SAMPLES
        x_local = a * math.cos(t) - a * e
        y_local = b * math.sin(t)
        cosw = math.cos(peri_angle)
        sinw = math.sin(peri_angle)
        p = pygame.Vector2(x_local * cosw - y_local * sinw, x_local * sinw + y_local * cosw)
        points.append(p)

    return {"ellipse_points": points, "apogee": apogee, "semi_major": a, "ecc": e, "center": center}


def world_to_screen(world: pygame.Vector2, camera: pygame.Vector2, scale: float) -> pygame.Vector2:
    shifted = (world - camera) * scale
    return CENTER + pygame.Vector2(shifted.x, -shifted.y)


def draw_glow(screen: pygame.Surface, pos: pygame.Vector2, power: float):
    # visually proportional to radiation level relative to nominal power
    ratio = max(power / P0, 1e-4)
    radius = max(3, min(28, int(3 + 8 * math.sqrt(ratio))))
    for r in range(radius, 0, -1):
        alpha = int(min(220, 90 + 45 * math.log10(1 + ratio * 8)) * (r / radius) ** 2)
        surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(surf, (140, 190, 255, alpha), (r, r), r)
        screen.blit(surf, (pos.x - r, pos.y - r))


def build_bodies() -> list[StellarBody]:
    bodies = [
        StellarBody("Sun", pygame.Vector2(0, 0), 1.989e30, 9.5e9, (255, 220, 140), 0, 0, 0, 0, False),
        StellarBody("Mercury", pygame.Vector2(), 3.30e23, 2.2e9, (186, 143, 109), 30, 0.387 * AU, 88.0, 0.0),
        StellarBody("Venus", pygame.Vector2(), 4.87e24, 3.2e9, (219, 184, 130), 70, 0.723 * AU, 224.7, 0.5),
        StellarBody("Earth", pygame.Vector2(), 5.97e24, 3.4e9, (120, 160, 255), 80, 1.0 * AU, 365.25, 1.1),
        StellarBody("Mars", pygame.Vector2(), 6.42e23, 2.6e9, (240, 130, 90), 40, 1.524 * AU, 687.0, 0.8),
    ]

    for i in range(120):
        orbit_radius = random.uniform(2.2 * AU, 3.2 * AU)
        period_days = (orbit_radius / AU) ** 1.5 * 365.25
        asteroid_radius = random.uniform(5_000.0, 40_000.0)  # 5-40 km radius
        asteroid_mass = (4 / 3) * math.pi * (asteroid_radius ** 3) * 2200.0
        bodies.append(
            StellarBody(
                f"Ast{i}",
                pygame.Vector2(),
                asteroid_mass,
                asteroid_radius,
                (155, 155, 155),
                HARVEST_MAX_FLOW,
                orbit_radius,
                period_days,
                random.uniform(0, math.tau),
                False,
            )
        )
    return bodies


def update_orbits(sim_time_days: float, bodies: list[StellarBody]):
    for body in bodies[1:]:
        phase = mean_anomaly_phase(body.orbit_period_days) + body.phase_offset + math.tau * sim_time_days / body.orbit_period_days
        body.pos = pygame.Vector2(math.cos(phase), math.sin(phase)) * body.orbit_radius


def draw_orbit_paths(screen: pygame.Surface, bodies: list[StellarBody], camera: pygame.Vector2, scale: float):
    sun_screen = world_to_screen(pygame.Vector2(0, 0), camera, scale)
    for body in bodies[1:]:
        orbit_radius_px = int(body.orbit_radius * scale)
        if orbit_radius_px < 2:
            continue
        color = (70, 100, 130) if body.is_planet else (48, 58, 72)
        pygame.draw.circle(screen, color, sun_screen, orbit_radius_px, 1)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Blackhole Drive Solar Navigator")
    font = pygame.font.SysFont("consolas", 18)
    clock = pygame.time.Clock()

    bodies = build_bodies()
    ship = Ship()
    sim_speed_idx = 0
    camera = ship.pos.copy()
    scale = 1.7e-9
    sim_time_days = 0.0
    debris_pos = None
    ship_trail: list[pygame.Vector2] = []

    hour_box = pygame.Rect(20, HEIGHT - 54, 140, 34)
    forward_btn = pygame.Rect(170, HEIGHT - 54, 120, 34)
    hour_input = "24"
    hour_input_active = False
    jump_seconds_remaining = 0.0
    jump_duration_real = 5.0
    jump_rate_seconds_per_real = 0.0
    jump_prev_speed_idx = 0
    alert_timer = 0.0
    alert_text = ""
    shockwave_origin: pygame.Vector2 | None = None
    shockwave_age = 0.0
    bh_free_pos = ship.pos.copy()
    bh_free_vel = ship.vel.copy()
    reset_btn = pygame.Rect(300, HEIGHT - 54, 150, 34)
    follow_camera = True
    dragging_view = False
    score_time = 0.0

    running = True
    while running:
        dt_real = clock.tick(60) / 1000.0
        dt_seconds = dt_real * TIME_STEPS[sim_speed_idx]
        if jump_seconds_remaining > 0:
            portion = min(jump_seconds_remaining, jump_rate_seconds_per_real * dt_real)
            dt_seconds = portion
            jump_seconds_remaining -= portion
            if jump_seconds_remaining <= 0:
                jump_seconds_remaining = 0.0
                sim_speed_idx = jump_prev_speed_idx

        sim_time_days += dt_seconds / 86400.0
        update_orbits(sim_time_days, bodies)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    hour_input_active = hour_box.collidepoint(event.pos)
                    if forward_btn.collidepoint(event.pos):
                        try:
                            hours = float(hour_input)
                            if hours > 0:
                                jump_seconds_remaining = hours * 3600
                                jump_rate_seconds_per_real = jump_seconds_remaining / max(jump_duration_real, 0.1)
                                jump_prev_speed_idx = sim_speed_idx
                                ship.aligned_for_thrust = False
                                ship.blackhole.feed_massflow = 0.0
                        except ValueError:
                            pass
                    if (not ship.blackhole.exists) and reset_btn.collidepoint(event.pos):
                        ship = Ship()
                        sim_speed_idx = 0
                        camera = ship.pos.copy()
                        scale = 1.7e-9
                        sim_time_days = 0.0
                        debris_pos = None
                        ship_trail = []
                        hour_input = "24"
                        hour_input_active = False
                        jump_seconds_remaining = 0.0
                        jump_rate_seconds_per_real = 0.0
                        jump_prev_speed_idx = 0
                        alert_timer = 0.0
                        alert_text = ""
                        shockwave_origin = None
                        shockwave_age = 0.0
                        bh_free_pos = ship.pos.copy()
                        bh_free_vel = ship.vel.copy()
                        follow_camera = True
                        dragging_view = False
                        score_time = 0.0
                        bodies = build_bodies()
                elif event.button == 2:
                    dragging_view = True
                    follow_camera = False
                elif event.button == 3:
                    follow_camera = True
                elif event.button == 4:
                    scale = min(1e-7, scale * 1.15)
                elif event.button == 5:
                    scale = max(1e-11, scale / 1.15)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                dragging_view = False
            elif event.type == pygame.MOUSEMOTION and dragging_view:
                dx, dy = event.rel
                camera -= pygame.Vector2(dx / scale, -dy / scale)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif hour_input_active:
                    if event.key == pygame.K_BACKSPACE:
                        hour_input = hour_input[:-1]
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        hour_input_active = False
                    elif event.unicode in "0123456789.":
                        hour_input += event.unicode
                elif event.key == pygame.K_SPACE and not ship.destroyed:
                    if jump_seconds_remaining > 0:
                        alert_timer = 1.8
                        alert_text = "Cannot toggle thrust during forward-jump."
                    elif TIME_STEPS[sim_speed_idx] > 100:
                        alert_timer = 1.8
                        alert_text = "Slow down to x100 or lower to enable thrust."
                    else:
                        ship.aligned_for_thrust = not ship.aligned_for_thrust
                elif event.key == pygame.K_y:
                    jump_seconds_remaining = 0.0
                    jump_rate_seconds_per_real = 0.0
                    sim_speed_idx = 0
                elif event.key == pygame.K_u:
                    jump_seconds_remaining = 0.0
                    jump_rate_seconds_per_real = 0.0
                    sim_speed_idx = 1
                elif event.key == pygame.K_i:
                    jump_seconds_remaining = 0.0
                    jump_rate_seconds_per_real = 0.0
                    sim_speed_idx = 2
                elif event.key == pygame.K_o:
                    jump_seconds_remaining = 0.0
                    jump_rate_seconds_per_real = 0.0
                    sim_speed_idx = 3
                elif event.key == pygame.K_p:
                    jump_seconds_remaining = 0.0
                    jump_rate_seconds_per_real = 0.0
                    sim_speed_idx = 4

        keys = pygame.key.get_pressed()
        ship_acc_g = 0.0
        if jump_seconds_remaining > 0 and (ship.aligned_for_thrust or ship.blackhole.feed_massflow > 0):
            ship.aligned_for_thrust = False
            ship.blackhole.feed_massflow = 0.0
            alert_timer = 1.5
            alert_text = "Thrust/feed disabled during forward-jump."
        if TIME_STEPS[sim_speed_idx] > 100 and (ship.aligned_for_thrust or ship.blackhole.feed_massflow > 0):
            ship.aligned_for_thrust = False
            ship.blackhole.feed_massflow = 0.0
            alert_timer = 1.8
            alert_text = "Thrust/feed disabled above x100. Slow down time to use propulsion."
        if alert_timer > 0:
            alert_timer = max(0.0, alert_timer - dt_real)
        if not ship.destroyed:
            if keys[pygame.K_a]:
                ship.heading += 0.9 * dt_seconds
            if keys[pygame.K_d]:
                ship.heading -= 0.9 * dt_seconds
            if TIME_STEPS[sim_speed_idx] <= 100 and jump_seconds_remaining <= 0:
                if keys[pygame.K_w]:
                    ship.blackhole.feed_massflow = min(100.0, ship.blackhole.feed_massflow + 20 * dt_seconds)
                if keys[pygame.K_s]:
                    ship.blackhole.feed_massflow = max(0.0, ship.blackhole.feed_massflow - 20 * dt_seconds)
            elif keys[pygame.K_w] or keys[pygame.K_s]:
                alert_timer = 1.2
                alert_text = "Mass feed locked (slow time / stop forward-jump)."

            if keys[pygame.K_c]:
                in_range_asteroids = [
                    b for b in bodies[1:]
                    if (not b.is_planet) and ship.pos.distance_to(b.pos) <= HARVEST_INFLUENCE_RADIUS
                ]
                if in_range_asteroids and ship.cargo_mass < ship.max_cargo:
                    mined = min(HARVEST_MAX_FLOW * dt_seconds, ship.max_cargo - ship.cargo_mass)
                    ship.cargo_mass += mined
                elif not in_range_asteroids:
                    alert_timer = 1.0
                    alert_text = "No asteroid inside 0.1 AU influence sphere for harvesting."

            consumed, _ = ship.blackhole.update(dt_seconds, ship.cargo_mass)
            ship.cargo_mass -= consumed

            acc = gravity_accel(ship.pos, bodies)
            if ship.aligned_for_thrust:
                thrust_force = ship.blackhole.thrust
                thrust_acc = pygame.Vector2(math.cos(ship.heading), math.sin(ship.heading)) * (thrust_force / ship.total_mass())
                acc += thrust_acc

            ship_acc_g = acc.length() / G0
            score_time += dt_seconds
            ship.vel += acc * dt_seconds
            ship.pos += ship.vel * dt_seconds
            bh_free_pos = ship.pos.copy()
            bh_free_vel = ship.vel.copy()
            ship_trail.append(ship.pos.copy())
            if len(ship_trail) > MAX_SHIP_TRAIL_POINTS:
                ship_trail.pop(0)

            if ship.blackhole.irradiation > 2 * ship.blackhole.nominal_limit:
                ship.destroyed = True
                ship.blackhole.failed = True
                debris_pos = ship.pos.copy()
            for body in bodies:
                if ship.pos.distance_to(body.pos) < body.radius * 0.9:
                    ship.destroyed = True
                    debris_pos = ship.pos.copy()
        else:
            ship.blackhole.update(dt_seconds, 0.0)
            if ship.blackhole.exists:
                bh_acc = gravity_accel(bh_free_pos, bodies)
                bh_free_vel += bh_acc * dt_seconds
                bh_free_pos += bh_free_vel * dt_seconds

        if (not ship.blackhole.exists) and shockwave_origin is None:
            shockwave_origin = bh_free_pos.copy()
            ship.blackhole.feed_massflow = 0.0
            ship.aligned_for_thrust = False

        if shockwave_origin is not None:
            shockwave_age += dt_seconds

        mu_sun = G * bodies[0].mass
        orbit_geom = compute_orbit_geometry(ship.pos, ship.vel, mu_sun) if not ship.destroyed else None
        camera_target = ship.pos if not ship.destroyed else (bh_free_pos if ship.blackhole.exists else debris_pos)
        if follow_camera and (not dragging_view):
            camera = camera.lerp(camera_target, 0.07)

        screen.fill((8, 12, 24))
        draw_orbit_paths(screen, bodies, camera, scale)

        if len(ship_trail) > 2:
            trail_pts = [world_to_screen(p, camera, scale) for p in ship_trail]
            pygame.draw.lines(screen, (90, 235, 255), False, trail_pts, 1)

        if orbit_geom and len(orbit_geom["ellipse_points"]) > 2:
            pred_pts = [world_to_screen(p, camera, scale) for p in orbit_geom["ellipse_points"]]
            pygame.draw.lines(screen, (255, 240, 120), True, pred_pts, 1)
            apo = world_to_screen(orbit_geom["apogee"], camera, scale)
            cross = 6
            pygame.draw.line(screen, (255, 120, 120), (apo.x - cross, apo.y - cross), (apo.x + cross, apo.y + cross), 2)
            pygame.draw.line(screen, (255, 120, 120), (apo.x - cross, apo.y + cross), (apo.x + cross, apo.y - cross), 2)

        for body in bodies:
            pos = world_to_screen(body.pos, camera, scale)
            rad = max(2, int(body.radius * scale))
            pygame.draw.circle(screen, body.color, pos, rad)

        if not ship.destroyed:
            ship_screen = world_to_screen(ship.pos, camera, scale)
            nose = ship_screen + pygame.Vector2(math.cos(ship.heading), -math.sin(ship.heading)) * 14
            left = ship_screen + pygame.Vector2(math.cos(ship.heading + 2.5), -math.sin(ship.heading + 2.5)) * 8
            right = ship_screen + pygame.Vector2(math.cos(ship.heading - 2.5), -math.sin(ship.heading - 2.5)) * 8
            pygame.draw.polygon(screen, (220, 240, 255), [nose, left, right])

            core = ship_screen + pygame.Vector2(-math.cos(ship.heading), math.sin(ship.heading)) * 4
            if ship.blackhole.exists:
                draw_glow(screen, core, ship.blackhole.irradiation)

                beam_len = 90
                cone_half_angle = math.radians(3)
                base_dir_angle = -ship.heading
                spread = math.radians(175) if ship.aligned_for_thrust else math.radians(90)
                for sign in (-1, 1):
                    center_angle = base_dir_angle + sign * spread
                    left_angle = center_angle - cone_half_angle
                    right_angle = center_angle + cone_half_angle
                    p_tip = core + pygame.Vector2(math.cos(center_angle), math.sin(center_angle)) * beam_len
                    p_left = core + pygame.Vector2(math.cos(left_angle), math.sin(left_angle)) * beam_len
                    p_right = core + pygame.Vector2(math.cos(right_angle), math.sin(right_angle)) * beam_len
                    pygame.draw.polygon(screen, (95, 190, 255), [core, p_left, p_tip, p_right])
                    pygame.draw.line(screen, (135, 225, 255), core, p_tip, 1)
        elif debris_pos:
            debris_screen = world_to_screen(debris_pos, camera, scale)
            pygame.draw.circle(screen, (180, 70, 70), debris_screen, 7)
            if ship.blackhole.exists:
                bh_screen = world_to_screen(bh_free_pos, camera, scale)
                draw_glow(screen, bh_screen, ship.blackhole.irradiation)
        if not ship.destroyed:
            ship_screen = world_to_screen(ship.pos, camera, scale)
            influence_px = max(2, int(HARVEST_INFLUENCE_RADIUS * scale))
            pygame.draw.circle(screen, (120, 180, 120), ship_screen, influence_px, 1)

        hud = [
            "A/D rotate W/S massflow SPACE beam C harvest",
            "Time speeds Y/U/I/O/P => 1x,10x,100x,10000x,1000000x",
            f"Speed: x{TIME_STEPS[sim_speed_idx]:.0f}  Jump left: {jump_seconds_remaining/3600:.2f} h",
            f"Cargo: {ship.cargo_mass:,.1f} kg  Feed: {ship.blackhole.feed_massflow:,.1f} kg/s",
            f"Acceleration: {ship_acc_g:,.3f} g",
            f"Radiation: {ship.blackhole.irradiation:,.3e} W  Ratio: {ship.blackhole.irradiation_ratio:,.3f}x nominal",
            f"BH mass: {ship.blackhole.mass:,.3e} kg  Hawking loss: {ship.blackhole.hawking_mass_loss_rate:,.3f} kg/s (sat)",
            f"Time to 2x power unfed: {ship.blackhole.time_to_2x_power_if_unfed/86400:.2f} days",
            f"Drive mode: {'THRUST' if ship.aligned_for_thrust else 'CANCEL'}",
            f"Score (alive time): {score_time/3600:.2f} h",
            "Green circle = 0.1 AU harvest influence | Hold C for 1 ton/s mining",
            f"Yellow line = osculating ellipse ({ORBIT_SHAPE_SAMPLES} samples)",
            "Red cross = predicted apogee from current state vector",
        ]
        if ship.blackhole.irradiation > ship.blackhole.nominal_limit:
            hud.append("WARNING: irradiation above nominal! >2x destroys ship")
        if ship.destroyed:
            hud.append("SHIP DESTROYED - black hole keeps evaporating")
            hud.append(f"Final score: {score_time/3600:.2f} h alive")

        for i, line in enumerate(hud):
            screen.blit(font.render(line, True, (210, 220, 240)), (20, 20 + 22 * i))

        pygame.draw.rect(screen, (36, 42, 60), hour_box, border_radius=4)
        pygame.draw.rect(screen, (110, 190, 255) if hour_input_active else (95, 105, 130), hour_box, 2, border_radius=4)
        pygame.draw.rect(screen, (68, 120, 78), forward_btn, border_radius=4)
        pygame.draw.rect(screen, (130, 220, 150), forward_btn, 2, border_radius=4)
        screen.blit(font.render(hour_input or "0", True, (240, 245, 255)), (hour_box.x + 8, hour_box.y + 8))
        screen.blit(font.render("hours", True, (170, 180, 205)), (hour_box.x + 88, hour_box.y - 20))
        screen.blit(font.render("Forward", True, (240, 255, 240)), (forward_btn.x + 18, forward_btn.y + 8))

        if not ship.blackhole.exists:
            pygame.draw.rect(screen, (90, 60, 60), reset_btn, border_radius=4)
            pygame.draw.rect(screen, (255, 170, 170), reset_btn, 2, border_radius=4)
            screen.blit(font.render("Reset Simulation", True, (255, 235, 235)), (reset_btn.x + 10, reset_btn.y + 8))

        if alert_timer > 0:
            warn_surf = font.render(alert_text, True, (255, 210, 120))
            warn_bg = pygame.Rect(20, HEIGHT - 96, warn_surf.get_width() + 16, 30)
            pygame.draw.rect(screen, (50, 36, 18), warn_bg, border_radius=4)
            pygame.draw.rect(screen, (220, 150, 70), warn_bg, 2, border_radius=4)
            screen.blit(warn_surf, (warn_bg.x + 8, warn_bg.y + 6))

        if shockwave_origin is not None:
            shock_screen = world_to_screen(shockwave_origin, camera, scale)
            radius_px = int(max(1, shockwave_age * C * scale * SHOCKWAVE_VISUAL_SCALE))
            if radius_px > 1 and radius_px < 100000:
                pygame.draw.circle(screen, (255, 220, 180), shock_screen, radius_px, 2)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()