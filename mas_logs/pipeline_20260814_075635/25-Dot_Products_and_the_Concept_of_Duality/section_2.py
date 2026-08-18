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
        lecture_lines = [
            "Duality turns a vector into a machine.",
            "Fix one vector to create a scalar function.",
            "Input vector maps to a single output.",
            "Like a caster creating a vector's shadow.",
            "Shadow length is the function's output value."
        ]
        self.setup_layout("The Duality Shift: From Vectors to Functions", lecture_lines)
        
        # Create objects
        plane = NumberPlane(x_range=[-3, 3], y_range=[-3, 3]).scale(0.5)
        # Apply fix for issue 24/36
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.8)
        
        v = Vector([1.5, 1], color=YELLOW)
        v_label = MathTex(r"\vec{v}", color=YELLOW)
        caster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/caster.svg", color=BLUE)
        shadow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/shadow.svg", color=RED)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(plane))
        # Apply fix for issue 25/37
        self.place_at_grid(v, 'C3', scale_factor=0.9)
        self.place_at_grid(v_label, 'C4', scale_factor=0.8)
        self.place_at_grid(caster, 'A2', scale_factor=0.5)
        self.play(GrowArrow(v), Write(v_label), FadeIn(caster))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        line = Line([-2, -1.33, 0], [2, 1.33, 0], color=BLUE).scale(0.5).move_to(plane.get_center())
        self.play(Create(line))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        dot = Dot(color=GREEN)
        self.place_at_grid(dot, 'B3')
        self.play(FadeIn(dot))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        projection = DashedLine(v.get_end(), [0.8, 0.5, 0], color=ORANGE)
        self.place_at_grid(shadow, 'E5', scale_factor=0.5)
        self.play(Create(projection), FadeIn(shadow))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        val_text = MathTex(r"f(\vec{v}) = \text{shadow length}", font_size=20, color=RED)
        # Apply fix for issue 26/38
        self.place_at_grid(val_text, 'F3', scale_factor=0.7)
        self.play(Write(val_text))
        
        self.wait(2)
