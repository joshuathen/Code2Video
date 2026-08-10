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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Distributions model how data outcomes are spread.",
            "A uniform distribution shows all outcomes are equal.",
            "Most real-world data is not uniform or simple."
        ]
        self.setup_layout("Prerequisite: The Concept of Distribution", lecture_lines)
        
        # Data points for visualization
        dots = VGroup(*[Dot(color=WHITE) for _ in range(50)])
        self.place_in_area(dots, 'B3', 'E6', scale_factor=0.9)
        
        # Labels
        bar_label = Text("Equal Probability", font_size=18, color="#FF5733")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        self.place_at_grid(bar_label, 'B4', scale_factor=1.0)
        self.play(FadeIn(bar_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        mean_dot = Dot(color="#33FF57", radius=0.2)
        mean_label = Text("Real Data", font_size=18, color="#33FF57")
        self.place_at_grid(mean_dot, 'D4', scale_factor=1.2)
        self.place_in_area(mean_label, 'F1', 'F6', scale_factor=0.8)
        self.play(FadeIn(mean_dot), FadeIn(mean_label))
        self.wait(2)
