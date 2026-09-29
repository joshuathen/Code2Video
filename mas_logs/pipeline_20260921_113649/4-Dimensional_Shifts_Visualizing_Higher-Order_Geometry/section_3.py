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
        lecture_lines = [
            "Points make lines, lines make squares.",
            "Squares move to form 3D cubes.",
            "Cubes shift into 4D as tesseracts.",
            "The tesseract is a 4D hypercube.",
            "It rotates through the fourth dimension."
        ]
        self.setup_layout("The Geometry of the Tesseract", lecture_lines)
        
        # Load asset for 3D visualization
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg
        cube_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        cube_svg.set_color("#FF00FF")
        
        # Positioning according to VideoCritic feedback (D4, scale_factor=0.4)
        self.place_at_grid(cube_svg, "D4", scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(FadeIn(cube_svg))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Using the same asset for Tesseract representation
        tesseract_wire = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        tesseract_wire.set_color("#00FF00")
        self.place_at_grid(tesseract_wire, "D4", scale_factor=0.6)
        self.play(Transform(cube_svg, tesseract_wire))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF8800"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        self.play(Rotate(cube_svg, angle=PI/4))
        self.wait(2)
