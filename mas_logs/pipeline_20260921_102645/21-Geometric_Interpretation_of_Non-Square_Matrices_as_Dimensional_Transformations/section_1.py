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
        self.setup_layout("Prerequisite: The Concept of Basis Vectors", [
            "Matrices represent transformations of coordinate systems.",
            "Basis vectors define the transformed space.",
            "Square matrices map space to itself."
        ])
        
        # Add background grid asset
        grid_image = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_image, 'B2', 'E5', scale_factor=0.8)
        self.add(grid_image)

        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": True})
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.8)
        self.add(axes)

        i_hat = Vector(RIGHT, color="#ADD8E6")
        j_hat = Vector(UP, color="#ADD8E6")
        
        i_label = Text("i-hat", font_size=20, color=WHITE)
        j_label = Text("j-hat", font_size=20, color=WHITE)
        
        self.place_at_grid(i_hat, 'D5', scale_factor=0.6)
        self.place_at_grid(j_hat, 'C4', scale_factor=0.6)
        
        # Setup static objects
        self.add(i_hat, j_hat)
        self.place_at_grid(i_label, 'E6', scale_factor=0.7)
        self.place_at_grid(j_label, 'A4', scale_factor=0.7)
        self.add(i_label, j_label)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        # Show span
        square = Polygon(ORIGIN, RIGHT, RIGHT+UP, UP, color="#D3D3D3", fill_opacity=0.3)
        self.place_in_area(square, 'B2', 'E5', scale_factor=0.8)
        self.add(square)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#ADD8E6"))
        self.play(i_hat.animate.put_start_and_end_on(ORIGIN, RIGHT + 0.5*UP),
                  j_hat.animate.put_start_and_end_on(ORIGIN, -0.5*RIGHT + UP))
        self.wait(2)
