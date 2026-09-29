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
        self.setup_layout("The Core Logic: The 'y-is-a-function' Rule", 
                          ["Treat y as a function of x.", 
                           "Differentiating y requires the Chain Rule.", 
                           "Multiply by dy/dx for every y term."])
        
        # === Animation for Lecture Line 1 ===
        def_text = Text("y = f(x)", font_size=36, color="#FFFFFF")
        self.place_at_grid(def_text, 'B1', scale_factor=0.85)
        self.play(Write(def_text))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using a group to hold formulas as suggested by feedback
        formula_group = VGroup(MathTex("d/dx [y^3]", color="#FF0000"), 
                               MathTex("=", color="#FF0000"), 
                               MathTex("3y^2 \\cdot (dy/dx)", color="#FF0000"))
        formula_group.arrange(RIGHT)
        self.place_in_area(formula_group, 'C2', 'E4', scale_factor=0.75)
        
        self.play(Write(formula_group))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"color": "#CCCCCC"}).scale(0.5)
        self.place_in_area(axes, 'C3', 'F5', scale_factor=0.9)
        self.play(Create(axes))
        
        # Highlight dy/dx alert
        dy_dx_alert = Text("dy/dx alert!", color="#00FF00", font_size=24)
        self.place_at_grid(dy_dx_alert, 'A6')
        self.play(FadeIn(dy_dx_alert))
        self.play(Indicate(formula_group[2]))
        
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
