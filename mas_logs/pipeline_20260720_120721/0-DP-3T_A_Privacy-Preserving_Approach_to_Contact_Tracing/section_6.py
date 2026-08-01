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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "DP-3T is a decentralized system.",
            "It protects user identity effectively.",
            "It's a model for future tracing apps.",
        ]
        self.setup_layout("DP-3T in Action: Real-World Applications & Future", lecture_lines)
        
        # Define colors for lecture lines and corresponding animations
        color1 = "#FF6347" # Tomato
        color2 = "#4682B4" # SteelBlue
        color3 = "#3CB371" # MediumSeaGreen

        # === Animation for Lecture Line 1 ===
        # Lecture line: "DP-3T is a decentralized system."
        # Animation: "FadeIn the text 'Real-World Apps'"
        # Issue 37: real_world_apps_text (C3) obstructs lecture line 1.
        # Resolution: Moving real_world_apps_text to C4 to avoid obstruction.
        self.play(self.lecture[0].animate.set_color(color1))
        real_world_apps_text = Text("Real-World Apps", font_size=28, color=color1)
        self.place_at_grid(real_world_apps_text, 'C4', scale_factor=0.9)
        self.play(FadeIn(real_world_apps_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture line: "It protects user identity effectively."
        # Animation: "FadeIn the text 'DP-3T Advantages'"
        # Issue 38: secure_decentralized_text (D3) is too close to and slightly overlaps with lecture line 2.
        # Resolution: Using 'DP-3T Advantages' text from storyboard. Moving this text to D4 to avoid obstruction.
        self.play(self.lecture[1].animate.set_color(color2))
        dp_3t_advantages_text = Text("DP-3T Advantages", font_size=28, color=color2)
        self.place_at_grid(dp_3t_advantages_text, 'D4', scale_factor=0.9)
        self.play(FadeIn(dp_3t_advantages_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture line: "It's a model for future tracing apps."
        # Animation: "FadeIn the text 'Future Potential'"
        # Issue 39: privacy_public_health_text (E3) is too close to and overlaps with lecture line 3.
        # Resolution: Using 'Future Potential' text from storyboard. Moving this text to E4 to avoid obstruction.
        self.play(self.lecture[2].animate.set_color(color3))
        future_potential_text = Text("Future Potential", font_size=28, color=color3)
        self.place_at_grid(future_potential_text, 'E4', scale_factor=0.8)
        self.play(FadeIn(future_potential_text))
        self.wait(1)

        # Cleanup: Fade out all animation texts at the end of the section
        self.play(
            FadeOut(real_world_apps_text),
            FadeOut(dp_3t_advantages_text),
            FadeOut(future_potential_text)
        )
        self.wait(1)
