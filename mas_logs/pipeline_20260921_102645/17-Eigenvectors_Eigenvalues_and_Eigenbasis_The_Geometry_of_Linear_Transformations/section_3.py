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
        self.setup_layout("Visualizing Eigenbasis", [
            "Eigenvectors can form a coordinate basis.",
            "The matrix becomes diagonal in this basis.",
            "Complex calculations simplify to independent scaling."
        ])
        
        # Define objects
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg]
        grid_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        plane = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        v1 = Vector([1, 1], color=WHITE)
        v2 = Vector([-1, 1], color=WHITE)
        
        group = VGroup(grid_bg, plane, v1, v2)
        self.place_in_area(group, 'C2', 'F5', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(grid_bg), Create(plane), GrowArrow(v1), GrowArrow(v2))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform to show diagonal/scaling
        self.play(
            plane.animate.stretch(1.5, 0).stretch(0.5, 1),
            v1.animate.set_length(2).set_color(BLUE),
            v2.animate.set_length(1).set_color(BLUE)
        )
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Use ruler to measure
        self.place_at_grid(ruler, 'B3', scale_factor=0.5)
        self.play(FadeIn(ruler))
        v1_highlight = v1.copy().set_color("#32CD32")
        v2_highlight = v2.copy().set_color("#32CD32")
        self.play(FadeIn(v1_highlight), FadeIn(v2_highlight))
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(2)
