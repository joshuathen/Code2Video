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
        self.setup_layout("Mapping Dimensions: 1D to 2D to 3D", [
            "Sweep a point to create a 1D line.",
            "Sweep a line to create a 2D square.",
            "Sweep a square to build a 3D cube."
        ])
        
        # Load Assets
        point_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg")
        line_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/line.svg")
        square_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        
        # Labels
        lbl_point = Text("Point", font_size=20, color=WHITE)
        lbl_line = Text("1D Line", font_size=20, color="#FF00FF")
        lbl_square = Text("2D Square", font_size=20, color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_at_grid(point_img, 'B4', scale_factor=0.5)
        self.place_at_grid(lbl_point, 'B5', scale_factor=0.5) # Label next to object
        self.play(FadeIn(point_img), Write(lbl_point))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.place_in_area(line_img, 'B3', 'B5', scale_factor=0.6)
        self.place_at_grid(lbl_line, 'C5', scale_factor=0.5)
        self.play(ReplacementTransform(point_img, line_img), FadeOut(lbl_point), Write(lbl_line))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_in_area(square_img, 'B2', 'E5', scale_factor=0.7)
        self.place_at_grid(lbl_square, 'E5', scale_factor=0.5)
        
        cube = Cube(side_length=1.5, fill_opacity=0.3, color=RED)
        self.place_in_area(cube, 'D3', 'F5', scale_factor=0.8)
        
        self.play(ReplacementTransform(line_img, square_img), FadeOut(lbl_line), Write(lbl_square))
        self.play(FadeIn(cube))
        self.wait(2)
