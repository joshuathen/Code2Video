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
            "Apply the sweep method to a 3D cube.",
            "Dragging it creates a 4D Tesseract.",
            "We visualize 4D objects through their shadows.",
            "The shadow projects into our 3D space.",
            "Complexity grows with every added spatial dimension."
        ]
        self.setup_layout("The Leap to 4D: The Tesseract", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        sq = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=WHITE)
        self.place_in_area(sq, 'B4', 'D6', scale_factor=0.9)
        self.play(FadeIn(sq))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        sq2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=WHITE).shift(UP * 0.3 + RIGHT * 0.3)
        self.play(FadeIn(sq2))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        tesseract = Cube(side_length=1.5, fill_opacity=0.2, stroke_color="#FFFF00").rotate(PI/6, axis=RIGHT+UP)
        self.place_at_grid(tesseract, 'C5', scale_factor=1.0)
        self.play(ReplacementTransform(VGroup(sq, sq2), tesseract))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(BLUE)
        self.play(Rotate(tesseract, angle=TAU/4, axis=UP))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE)
        icons = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color="#00FFFF").scale(0.3) for _ in range(8)])
        # Just creating some effect as a placeholder for the flash
        self.play(Flash(tesseract, color="#00FFFF", run_time=2))
