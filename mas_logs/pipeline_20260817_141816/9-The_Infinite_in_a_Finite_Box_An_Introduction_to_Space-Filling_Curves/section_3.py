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
            "Finite steps are just simple line segments.",
            "The limit is a 2D curve.",
            "It maps 1D to 2D continuously."
        ]
        self.setup_layout("Crossing the Finite-Infinite Bridge", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Show finite set of objects [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/objects.svg]
        objects = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/objects.svg", color=WHITE)
        self.place_at_grid(objects, 'B2', scale_factor=0.6)
        self.play(Create(objects))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Introduce Cantor's mapping [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg]
        bridge = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg", color="#FF33FF")
        self.place_at_grid(bridge, 'D4', scale_factor=0.7)
        self.play(Create(bridge))
        self.play(self.lecture[1].animate.set_color("#FF33FF"))

        # === Animation for Lecture Line 3 ===
        # Show mapping (1D segment to 2D square)
        line_to_square = Line(start=self.grid['B2'], end=self.grid['D4'], color="#FFFF33", stroke_width=2)
        label = Text("f: [0,1] -> [0,1]x[0,1]", font_size=20, color="#FFFF33")
        self.place_at_grid(label, 'C6', scale_factor=0.65)
        self.play(Create(line_to_square), Write(label))
        self.play(self.lecture[2].animate.set_color("#FFFF33"))
        
        self.wait(2)
