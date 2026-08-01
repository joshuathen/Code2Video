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
        self.setup_layout("The Chain Rule: Composition of Scaling", [
            "Chain rule is a composition of transformations.",
            "If first map scales by three, space expands.",
            "If second map scales by two, expansion continues.",
            "Scaling factors multiply through the composite function.",
            "Total scaling is the product of individual derivatives."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        def create_axis(label_text):
            line = Line(LEFT*2, RIGHT*2, color=WHITE)
            label = MathTex(label_text, color=WHITE).next_to(line, LEFT, buff=0.2)
            return VGroup(line, label)

        axis_x = create_axis("x")
        axis_u = create_axis("u")
        axis_y = create_axis("y")

        # Fixed axis_y placement (Issue 32)
        self.place_in_area(axis_x, "B1", "B6", scale_factor=0.8)
        self.place_in_area(axis_u, "D1", "D6", scale_factor=0.8)
        self.place_in_area(axis_y, "F1", "F6", scale_factor=0.8)

        self.play(Create(axis_x), Create(axis_u), Create(axis_y))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        
        center_x = axis_x[0].get_center()
        center_u = axis_u[0].get_center()
        
        seg_x = Line(center_x + LEFT*0.2, center_x + RIGHT*0.2, color="#00FFFF", stroke_width=6)
        seg_u = Line(center_u + LEFT*0.6, center_u + RIGHT*0.6, color="#00FFFF", stroke_width=6)
        
        map_arrow_1 = CurvedArrow(self.grid["B4"] + DOWN*0.2, self.grid["D4"] + UP*0.2, angle=-TAU/8, color="#00FFFF")
        label_g = MathTex("g'(x) = 3", color="#00FFFF", font_size=24)
        self.place_at_grid(label_g, "C5", scale_factor=1.0)

        self.play(Create(seg_x))
        self.play(ReplacementTransform(seg_x.copy(), seg_u), Create(map_arrow_1), Write(label_g))
        self.add(seg_x) 
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        
        center_y = axis_y[0].get_center()
        seg_y = Line(center_y + LEFT*1.2, center_y + RIGHT*1.2, color="#00FF00", stroke_width=6)
        
        map_arrow_2 = CurvedArrow(self.grid["D5"] + DOWN*0.2, self.grid["F5"] + UP*0.2, angle=-TAU/8, color="#00FF00")
        label_f = MathTex("f'(u) = 2", color="#00FF00", font_size=24)
        # Fixed label_f placement (Issue 34)
        self.place_at_grid(label_f, "E2", scale_factor=0.8)

        self.play(ReplacementTransform(seg_u.copy(), seg_y), Create(map_arrow_2), Write(label_f))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        
        total_arrow = CurvedArrow(self.grid["B1"] + LEFT*0.3, self.grid["F1"] + LEFT*0.3, angle=TAU/4, color="#FF00FF")
        label_total = MathTex("Total: \\times 6", color="#FF00FF", font_size=24)
        # Fixed label_total placement (Issue 33)
        self.place_at_grid(label_total, "D2", scale_factor=0.8)

        self.play(Create(total_arrow), Write(label_total))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        
        formula = MathTex(
            "(f(g(x)))' = f'(g(x)) \\cdot g'(x) = 2 \\cdot 3 = 6",
            color="#FF00FF", font_size=28
        )
        self.place_in_area(formula, "A1", "A6", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(2)
