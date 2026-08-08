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
        self.setup_layout("Defining Basis Vectors", [
            "A basis is a system of independent vectors.",
            "Standard basis vectors define our default grid.",
            "Custom basis vectors tilt and stretch the grid."
        ])

        # Setup standard grid
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], axis_config={"stroke_opacity": 0.3})
        self.place_in_area(grid, 'B2', 'E5', scale_factor=0.75)
        
        # Load asset: SVGMobject should be used for .svg files instead of ImageMobject
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_asset, 'B2', 'E5', scale_factor=0.5)
        
        self.add(grid, grid_asset)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        i_hat = Vector(RIGHT, color="#00FF00")
        j_hat = Vector(UP, color="#00FF00")
        self.place_at_grid(i_hat, 'D4', scale_factor=0.9)
        self.place_at_grid(j_hat, 'C3', scale_factor=1.0)
        self.play(Create(i_hat), Create(j_hat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Already showing standard vectors
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        b1 = Vector(RIGHT + UP, color="#FF00FF")
        b2 = Vector(-RIGHT + UP, color="#FF00FF")
        
        # Applying requested fixes
        self.place_at_grid(b1, 'D5', scale_factor=0.8)
        self.place_at_grid(b2, 'B3', scale_factor=0.8)
        
        dashed_line = DashedLine(i_hat.get_end(), b1.get_end(), color=WHITE)
        dashed_line2 = DashedLine(j_hat.get_end(), b2.get_end(), color=WHITE)
        
        self.play(Create(b1), Create(b2), Create(dashed_line), Create(dashed_line2))
        self.wait(2)
