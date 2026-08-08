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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Application", [
            "We only change our coordinate description.",
            "The vector remains physically unchanged.",
            "Grid lines rotate, labels update accordingly."
        ])
        
        # Assets
        prot_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg", color=WHITE)
        comp_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        glob_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg", color=WHITE)
        
        # Recapping Basis Formula
        formula = MathTex(r"[\mathbf{v}]_B = P^{-1} [\mathbf{v}]_A", color=WHITE)
        self.place_at_grid(formula, 'B4', scale_factor=1.0)
        self.place_at_grid(prot_icon, 'A4', scale_factor=0.3)
        
        # 2D Grid Representation for Rotation
        plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_opacity": 0.3})
        vec = Vector([1, 1], color=BLUE)
        grp = VGroup(plane, vec)
        self.place_in_area(grp, 'C4', 'F6', scale_factor=0.4)
        self.place_at_grid(comp_icon, 'C3', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(formula), FadeIn(prot_icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.play(Create(grp), FadeIn(comp_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(Rotate(plane, angle=PI/6), Rotate(vec, angle=PI/6))
        self.wait(1)
        
        final_text = Text("Perspective changes.", color=WHITE).scale(0.8)
        self.place_at_grid(final_text, 'E3', scale_factor=0.9)
        self.place_at_grid(glob_icon, 'E4', scale_factor=0.3)
        self.play(FadeIn(final_text), FadeIn(glob_icon))
        self.wait(2)
