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
            "Meet the Tesseract, a 4D hypercube.",
            "We visualize 4D by projecting into 3D.",
            "A 3D cube unfolds into a 2D cross.",
            "A Tesseract unfolds into 3D cubes.",
            "This reveals the nature of higher dimensions."
        ]
        self.setup_layout("The Dimensional Shift Puzzle: The Tesseract", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E74C3C"))
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg] as reference
        tesseract_wire = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        tesseract_wire.set_stroke(color="#E74C3C", width=2)
        self.place_at_grid(tesseract_wire, 'C4', scale_factor=0.8) # Adjusted per issue 27/38
        self.play(Create(tesseract_wire))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#3498DB"))
        # Visualization of projection logic
        projection_line = DashedLine(start=UP, end=DOWN, color=WHITE).scale(0.5)
        self.place_at_grid(projection_line, 'D5', scale_factor=0.7) # Adjusted per issue 28/38
        self.play(GrowFromCenter(projection_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg]
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        cube.set_fill(color="#3498DB", opacity=0.3)
        cube.set_stroke(color="#3498DB", width=2)
        self.place_at_grid(cube, 'B5', scale_factor=0.5) # Adjusted per issue 29/38
        self.play(Create(cube))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#F1C40F"))
        # Tesseract unfolding using cube assets
        cubes = VGroup(*[SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg").set_fill(color="#E74C3C", opacity=0.3).scale(0.3) for _ in range(8)])
        self.place_in_area(cubes, 'E2', 'F6', scale_factor=0.8)
        self.play(FadeIn(cubes))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#9B59B6"))
        self.play(Circumscribe(self.lecture))
        self.wait(2)
