from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialize attributes used in construct by calling setup_layout
        self.setup_layout(
            "The Digital Handshake: Proximity Exchange",
            [
                "Phones exchange random temporary IDs via Bluetooth beacons.",
                "Devices store only the IDs they have encountered locally.",
                "No location data or personal info is ever exchanged."
            ]
        )

        # Colors
        ALICE_COLOR = "#58D68D"
        BOB_COLOR = "#5DADE2"
        LOG_COLOR = "#BDC3C7"
        GPS_COLOR = "#E74C3C"
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        # Phones exchange random temporary IDs via Bluetooth beacons.
        self.lecture[0].set_color(ALICE_COLOR)

        # Alice's and Bob's Phone setup (with fallback to Rectangle if SVGMobject fails)
        try:
            alice_phone = SVGMobject(PHONE_ASSET, color=ALICE_COLOR, height=1.2)
            bob_phone = SVGMobject(PHONE_ASSET, color=BOB_COLOR, height=1.2)
        except:
            alice_phone = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=ALICE_COLOR)
            bob_phone = RoundedRectangle(corner_radius=0.1, height=1.2, width=0.7, color=BOB_COLOR)

        alice_label = Text("Alice", font_size=16, color=ALICE_COLOR)
        alice_group = VGroup(alice_phone, alice_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(alice_group, "B2")

        bob_label = Text("Bob", font_size=16, color=BOB_COLOR)
        bob_group = VGroup(bob_phone, bob_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(bob_group, "B5")

        self.play(FadeIn(alice_group), FadeIn(bob_group))

        # RPI packets
        rpi_alice_rect = RoundedRectangle(corner_radius=0.05, height=0.3, width=0.6, color=ALICE_COLOR, fill_opacity=0.5)
        rpi_alice_text = Text("RPI_A", font_size=12, color=WHITE).move_to(rpi_alice_rect.get_center())
        packet_alice = VGroup(rpi_alice_rect, rpi_alice_text)

        rpi_bob_rect = RoundedRectangle(corner_radius=0.05, height=0.3, width=0.6, color=BOB_COLOR, fill_opacity=0.5)
        rpi_bob_text = Text("RPI_B", font_size=12, color=WHITE).move_to(rpi_bob_rect.get_center())
        packet_bob = VGroup(rpi_bob_rect, rpi_bob_text)

        packet_alice.move_to(alice_group.get_center())
        packet_bob.move_to(bob_group.get_center())

        self.play(
            packet_alice.animate.move_to(bob_group.get_center()),
            packet_bob.animate.move_to(alice_group.get_center()),
            run_time=2
        )
        self.play(FadeOut(packet_alice), FadeOut(packet_bob))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Devices store only the IDs they have encountered locally.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(LOG_COLOR)

        # Bob's Local Log
        log_title = Text("Bob's Local Log", font_size=18, color=LOG_COLOR)
        log_box = Rectangle(height=1.5, width=2.5, color=LOG_COLOR)
        log_entry = Text("Encounter: RPI_A", font_size=14, color=WHITE)
        log_content = VGroup(log_title, log_entry).arrange(DOWN, buff=0.2)
        log_group = VGroup(log_box, log_content)
        
        self.place_in_area(log_group, "C4", "E6", scale_factor=0.85)

        self.play(Create(log_box), Write(log_title))
        self.play(FadeIn(log_entry, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # No location data or personal info is ever exchanged.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GPS_COLOR)

        # GPS Icon
        gps_circle = Circle(radius=0.3, color=GPS_COLOR)
        gps_dot = Dot(color=GPS_COLOR).move_to(gps_circle.get_center())
        gps_pin = Triangle(color=GPS_COLOR).scale(0.2).rotate(180*DEGREES).next_to(gps_circle, DOWN, buff=0)
        gps_icon = VGroup(gps_circle, gps_dot, gps_pin)
        gps_label = Text("GPS", font_size=16, color=GPS_COLOR).next_to(gps_icon, DOWN, buff=0.1)
        gps_total = VGroup(gps_icon, gps_label)
        
        self.place_at_grid(gps_total, "D2", scale_factor=0.8)

        # Red X
        red_x = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=RED, stroke_width=8),
            Line(UP+RIGHT, DOWN+LEFT, color=RED, stroke_width=8)
        ).scale(0.4).move_to(gps_total.get_center())

        self.play(FadeIn(gps_total))
        self.play(Create(red_x))
        self.wait(2)

        # Final state
        self.lecture[2].set_color(WHITE)
        self.wait(1)
