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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "Phase 1: The Digital Whisper (Broadcast)"
        lines = [
            "Alice and Bob's phones exchange anonymous temporary codes.",
            "Alice's phone broadcasts a random ID like XJ-9.",
            "Bob's phone listens and records this in his log.",
            "Alice's ID changes frequently to prevent tracking.",
            "Bob continues to log these nearby anonymous whispers."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        ALICE_COLOR = "#58D68D"
        BOB_COLOR = "#F5B041"
        PULSE_COLOR = "#5DADE2"
        
        # Mobjects setup
        # Alice Phone
        alice_phone = RoundedRectangle(corner_radius=0.1, height=1.4, width=0.8, color=ALICE_COLOR)
        alice_label = Text("Alice", font_size=20, color=ALICE_COLOR)
        alice_group = VGroup(alice_phone, alice_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(alice_group, "B2")
        
        # Bob Phone
        bob_phone = RoundedRectangle(corner_radius=0.1, height=1.4, width=0.8, color=BOB_COLOR)
        bob_label = Text("Bob", font_size=20, color=BOB_COLOR)
        bob_group = VGroup(bob_phone, bob_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(bob_group, "B5")
        
        # Visitor Log (Issue 39 fix: move to F5)
        log_rect = RoundedRectangle(corner_radius=0.05, height=1.6, width=2.2, color=WHITE)
        log_title = Text("Visitor Log", font_size=18, color=WHITE)
        log_vgroup = VGroup(log_title, log_rect).arrange(DOWN, buff=0.1)
        self.place_at_grid(log_vgroup, "F5", scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(PULSE_COLOR)
        self.play(FadeIn(alice_group), FadeIn(bob_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(ALICE_COLOR)
        
        # Pulse XJ-9
        p1_circ = Circle(radius=0.4, color=PULSE_COLOR, fill_opacity=0.3)
        p1_text = Text("XJ-9", font_size=16, color=WHITE)
        p1 = VGroup(p1_circ, p1_text)
        self.place_at_grid(p1, "B2")
        
        # BLE Text (Issue 38 fix: move to A4, scale 0.8)
        ble_text = Text("Bluetooth Low Energy", font_size=18, color=PULSE_COLOR)
        self.place_at_grid(ble_text, "A4", scale_factor=0.8)
        
        # Propagation effect
        prop = Circle(radius=0.1, color=PULSE_COLOR, stroke_width=2).move_to(self.grid["B2"])
        
        self.play(Write(ble_text), FadeIn(p1))
        self.play(prop.animate.scale(15).set_stroke(opacity=0), run_time=1.0)
        self.remove(prop)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BOB_COLOR)
        
        self.play(FadeIn(log_vgroup))
        self.play(p1.animate.move_to(self.grid["B5"]), run_time=1.2)
        
        entry1 = Text("XJ-9", font_size=16, color=PULSE_COLOR)
        entry1.move_to(log_rect.get_center() + UP * 0.4)
        
        self.play(FadeOut(p1), FadeIn(entry1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(RED)
        
        # Privacy Group (Issue 37 fix: move to D4, scale 0.7)
        privacy_vgroup = VGroup(
            Text("Name: Alice", font_size=16),
            Text("Location: (40.7, -74.0)", font_size=16)
        ).arrange(DOWN, buff=0.1)
        self.place_at_grid(privacy_vgroup, "D4", scale_factor=0.7)
        cross = Cross(privacy_vgroup, color=RED, stroke_width=4)
        
        # Pulse QK-4
        p2_circ = Circle(radius=0.4, color=PULSE_COLOR, fill_opacity=0.3)
        p2_text = Text("QK-4", font_size=16, color=WHITE)
        p2 = VGroup(p2_circ, p2_text)
        self.place_at_grid(p2, "B2")
        
        self.play(FadeIn(privacy_vgroup))
        self.play(Create(cross))
        self.play(FadeIn(p2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PULSE_COLOR)
        
        entry2 = Text("QK-4", font_size=16, color=PULSE_COLOR)
        entry2.next_to(entry1, DOWN, buff=0.1)
        
        self.play(p2.animate.move_to(self.grid["B5"]), run_time=1.2)
        self.play(FadeOut(p2), FadeIn(entry2))
        self.wait(2)
