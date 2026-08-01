from manim import *

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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup layout with specific DP-3T prerequisite text
        title = "Prerequisite: The One-Way Hash & Bluetooth RSSI"
        lines = [
            "Cryptographic hashes create unique, irreversible digital fingerprints.",
            "Bluetooth RSSI measures signal strength to estimate distance.",
            "These tools enable privacy-preserving proximity detection."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1: Cryptographic Hash ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # 1. Create Input Box - Fixed Position (Issue 41)
        input_box = VGroup(
            RoundedRectangle(corner_radius=0.1, width=2, height=1, color=WHITE),
            Text("Data", font_size=24)
        )
        self.place_at_grid(input_box, "A3", scale_factor=0.8)
        
        # 2. Create Gear
        gear_color = "#95a5a6"
        gear_core = Circle(radius=0.5, color=gear_color, fill_opacity=1)
        teeth = VGroup(*[
            Rectangle(width=0.2, height=0.2, color=gear_color, fill_opacity=1)
            .move_to(gear_core.get_center() + [0.6 * np.cos(TAU * i / 8), 0.6 * np.sin(TAU * i / 8), 0])
            .rotate(TAU * i / 8)
            for i in range(8)
        ])
        gear = VGroup(gear_core, teeth)
        self.place_at_grid(gear, "B3", scale_factor=0.8)

        # 3. Create Output Hash - Fixed Position and Scale (Issue 42)
        output_hash = Text("a7b2...6f8c", font="Monospace", font_size=24, color=GRAY)
        self.place_at_grid(output_hash, "D3", scale_factor=0.7)

        # Sequence Line 1
        self.play(FadeIn(input_box))
        self.wait(0.5)
        self.play(input_box.animate.move_to(gear.get_center()), gear.animate.rotate(PI), run_time=1)
        self.remove(input_box)
        self.play(FadeIn(output_hash), gear.animate.rotate(PI))
        self.wait(1)

        # Reversal Attempt (The "One-way" property)
        no_entry_color = "#e74c3c"
        cross = Cross(gear, stroke_color=no_entry_color, stroke_width=8)
        
        self.play(output_hash.animate.move_to(gear.get_center()), run_time=1)
        self.play(
            gear.animate.set_color(no_entry_color),
            Create(cross),
            run_time=0.5
        )
        self.wait(2)

        # Clean up Line 1
        self.play(
            FadeOut(gear), FadeOut(cross), FadeOut(output_hash),
            self.lecture[0].animate.set_color(WHITE)
        )

        # === Animation for Lecture Line 2: Bluetooth RSSI ===
        self.play(self.lecture[1].animate.set_color(BLUE))

        # Phone Icons - Fixed Position (Issue 43)
        def get_phone(label):
            body = RoundedRectangle(corner_radius=0.1, width=1, height=1.8, color=WHITE, fill_opacity=1, fill_color=BLACK)
            screen = Rectangle(width=0.8, height=1.4, color=WHITE).move_to(body.get_center())
            lbl = Text(label, font_size=18).move_to(screen.get_center())
            return VGroup(body, screen, lbl)

        phone_a = get_phone("Phone A")
        phone_b = get_phone("Phone B")
        self.place_at_grid(phone_a, "E2", scale_factor=0.7)
        self.place_at_grid(phone_b, "E5", scale_factor=0.7)

        # Bluetooth Expanding Rings
        def get_rings(source):
            rings = VGroup(*[Circle(color="#3498db", stroke_width=2) for _ in range(3)])
            for i, ring in enumerate(rings):
                ring.scale(0.1)
                ring.set_stroke(opacity=0)
                ring.time_offset = i * (1/3)
            
            def update_rings(m, dt):
                for ring in m:
                    if not hasattr(ring, "t"): ring.t = ring.time_offset
                    ring.t += dt * 0.5
                    if ring.t > 1: ring.t -= 1
                    
                    new_radius = 0.1 + ring.t * 1.5
                    ring.scale_to_fit_width(new_radius * 2)
                    ring.set_stroke(opacity=1 - ring.t)
                    ring.move_to(source.get_center())
            
            rings.add_updater(update_rings)
            return rings

        rings_a = get_rings(phone_a)
        rings_b = get_rings(phone_b)

        # RSSI Label
        rssi_val = DecimalNumber(-40, num_decimal_places=0, font_size=24, color=BLUE, mob_class=Text)
        rssi_text = Text("RSSI: ", font_size=24, color=BLUE)
        rssi_unit = Text(" dBm", font_size=24, color=BLUE)
        rssi_group = VGroup(rssi_text, rssi_val, rssi_unit).arrange(RIGHT, buff=0.1)
        
        # Link RSSI label to phone_b position
        rssi_group.add_updater(lambda m: m.next_to(phone_b, UP, buff=0.2))

        self.play(FadeIn(phone_a), FadeIn(phone_b), FadeIn(rings_a), FadeIn(rings_b), FadeIn(rssi_group))

        # Animation logic for RSSI decrease
        def update_rssi_val(m):
            dist = np.linalg.norm(phone_b.get_center() - phone_a.get_center())
            # Simplified RSSI formula for visual: starts at ~-55 at E5, drops as distance increases
            new_val = -40 - (dist - 1.0) * 10
            m.set_value(max(new_val, -95))

        rssi_val.add_updater(update_rssi_val)

        # Move phone B further away
        self.play(
            phone_b.animate.move_to(self.grid["E6"]),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3: Conclusion ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )
        
        # Pulse everything for emphasis
        self.play(
            phone_a.animate.scale(1.1),
            phone_b.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1
        )
        
        self.wait(2)
