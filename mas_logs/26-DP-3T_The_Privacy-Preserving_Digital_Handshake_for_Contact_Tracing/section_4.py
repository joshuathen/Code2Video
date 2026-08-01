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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Phase 2: The Digital Handshake", 
            [
                'Phones exchange rotating nicknames when users are nearby.', 
                'Each device stores only the IDs it has heard.', 
                'No location or personal data is ever recorded.', 
                'This "handshake" happens entirely in the background.', 
                'Your phone only remembers "who" it saw, not "where".'
            ]
        )

        # Alice and Bob Icons
        alice_color = "#FFD700"
        bob_color = "#87CEEB"
        
        alice_icon = VGroup(
            Circle(radius=0.4, color=alice_color, fill_opacity=0.3),
            Text("Alice", font_size=18, color=alice_color).shift(DOWN * 0.6)
        )
        bob_icon = VGroup(
            Circle(radius=0.4, color=bob_color, fill_opacity=0.3),
            Text("Bob", font_size=18, color=bob_color).shift(DOWN * 0.6)
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(alice_color)
        self.place_at_grid(alice_icon, "B1")
        self.place_at_grid(bob_icon, "B6")
        
        self.play(FadeIn(alice_icon), FadeIn(bob_icon))
        self.play(
            alice_icon.animate.move_to(self.grid["B3"]),
            bob_icon.animate.move_to(self.grid["B4"]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(WHITE) # Reset color or keep it active
        self.lecture[1].set_color("#FFFFFF")
        
        packet_a = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.4, width=1.0, color=WHITE, fill_opacity=0.8),
            Text("EphID_A", font_size=14, color=BLACK)
        )
        packet_b = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.4, width=1.0, color=WHITE, fill_opacity=0.8),
            Text("EphID_B", font_size=14, color=BLACK)
        )

        self.place_at_grid(packet_a, "B3", scale_factor=0.8)
        self.place_at_grid(packet_b, "B4", scale_factor=0.8)

        self.play(FadeIn(packet_a), FadeIn(packet_b))
        self.play(
            packet_a.animate.move_to(self.grid["B4"]),
            packet_b.animate.move_to(self.grid["B3"]),
            run_time=1.5
        )
        self.play(FadeOut(packet_a), FadeOut(packet_b))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        log_color = "#00FF00"
        log_title = Text("Local Interaction Log", font_size=20, color=log_color)
        log_box = Rectangle(height=2.5, width=3.5, color=log_color)
        log_content = VGroup(
            Text("- EphID_A  [2:00 PM]", font_size=16, color=WHITE),
            Text("- [No GPS Data]", font_size=16, color=RED)
        ).arrange(DOWN, aligned_edge=LEFT)
        
        log_group = VGroup(log_title, log_box, log_content).arrange(DOWN, buff=0.2)
        self.place_in_area(log_group, "D4", "F6", scale_factor=0.8)

        self.play(Create(log_box), Write(log_title))
        self.play(Write(log_content[0]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        # Bluetooth pulse animation
        pulse_circle = Circle(radius=0.1, color=BLUE, stroke_opacity=0.8).move_to(self.grid["B3"])
        self.add(pulse_circle)
        self.play(
            pulse_circle.animate.scale(10).set_stroke(opacity=0),
            run_time=1.5,
            rate_func=linear
        )
        self.remove(pulse_circle)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#87CEEB")
        
        # Emphasize the log and the "No GPS" part
        self.play(log_content[1].animate.scale(1.2).set_color(YELLOW), run_time=1)
        self.play(log_content[1].animate.scale(1/1.2).set_color(RED), run_time=1)
        
        # Final highlight of the Bob phone hearing Alice
        indication = SurroundingRectangle(log_content[0], color=YELLOW)
        self.play(Create(indication))
        self.wait(2)
        self.play(FadeOut(indication))
        self.wait(1)
