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
        lecture_lines = [
            "DP-3T uses Bluetooth for proximity.",
            "It generates random, rotating IDs.",
            "This prevents tracking of individuals."
        ]
        self.setup_layout("Introducing DP-3T: Decentralized, Privacy-Preserving", lecture_lines)

        # Define colors for lecture lines and animations
        color_1 = "#FF6347" # Tomato
        color_2 = "#4682B4" # SteelBlue
        color_3 = "#32CD32" # LimeGreen

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        dp3t_decentralized = Text("DP-3T: Decentralized", color=color_1, font_size=30)
        self.place_at_grid(dp3t_decentralized, 'C4', scale_factor=0.8)
        self.play(FadeIn(dp3t_decentralized))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_2))
        dp3t_privacy_preserving = Text("DP-3T: Privacy-Preserving", color=color_2, font_size=30)
        self.place_at_grid(dp3t_privacy_preserving, 'D4', scale_factor=0.8)
        self.play(FadeIn(dp3t_privacy_preserving))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_3))
        dp3t_solution = Text("DP-3T: Solution", color=color_3, font_size=30)
        self.place_at_grid(dp3t_solution, 'E5', scale_factor=0.8)
        self.play(FadeIn(dp3t_solution))
        self.wait(1)

        self.wait(2) # Final wait to keep everything on screen
