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
        self.setup_layout("Prerequisite: Discrete vs. Continuous Invariants", [
            "Categorize tools into discrete and continuous.",
            "Use parity and modular arithmetic for discrete.",
            "Use monotone functions or potentials for continuous."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        discrete_points = VGroup(*[Dot(self.grid[pos], color="#00FFFF") for pos in ["B2", "B4", "C3", "D2", "D4"]])
        
        # Using a VGroup to represent the discrete points set
        discrete_points_group = discrete_points
        self.place_at_grid(discrete_points_group, 'B3', scale_factor=0.9)
        self.play(FadeIn(discrete_points_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        continuous_line = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-1, 1], color="#FF00FF")
        
        # Fixing per VideoCritic (Issue 23, 38)
        self.place_in_area(continuous_line, 'D2', 'F5', scale_factor=0.8)
        self.play(FadeIn(continuous_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(
            *[p.animate.move_to(continuous_line.point_from_proportion((i+1)/6)) for i, p in enumerate(discrete_points)],
            run_time=2
        )
        self.wait(1)
