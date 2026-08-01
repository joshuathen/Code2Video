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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialize Layout
        title = "Integration: The Art of Slicing and Adding"
        lines = [
            "Integration calculates the total by summing up small parts.",
            "Imagine slicing the area under a curve into rectangles.",
            "Thinner rectangles provide a more precise total area.",
            "Summing infinite slivers reveals the exact whole.",
            "It turns a changing rate back into a total amount."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # "Integration calculates the total by summing up small parts."
        self.lecture[0].set_color("#00FF00")
        
        # Define Axes in area B2 to E5
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": "#FFFFFF"}
        )
        self.place_in_area(axes, "B2", "E5")
        
        # Curve: f(x) = 0.1 * x^2 + 0.5
        curve = axes.plot(lambda x: 0.1 * x**2 + 0.5, x_range=[0, 4], color="#00FF00")
        # Issue 29: Move curve_label to B5
        curve_label = MathTex("f(x)", color="#00FF00")
        self.place_at_grid(curve_label, "B5", scale_factor=0.7)
        
        self.play(Create(axes), Create(curve), FadeIn(curve_label))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # "Imagine slicing the area under a curve into rectangles."
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#00FF00")
        
        # Shade the area beneath it in a darker green (L008)
        area = axes.get_area(curve, x_range=[0, 4], color="#008000", opacity=0.5)
        # Issue 28: Move area_label to D4
        area_label = Text("Area", color="#008000")
        self.place_at_grid(area_label, "D4", scale_factor=0.6)
        
        self.play(FadeIn(area), FadeIn(area_label))
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # "Thinner rectangles provide a more precise total area."
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#FFFF00")
        
        # Overlay 5 thick yellow #FFFF00 rectangles
        rects_5 = axes.get_riemann_rectangles(
            curve, 
            x_range=[0, 4], 
            dx=0.8, 
            color="#FFFF00", 
            fill_opacity=0.4, 
            stroke_width=1,
            stroke_color="#FFFF00"
        )
        
        # Issue 30: Move sum_sym to area A2-A5
        sum_sym = MathTex(r"\sum f(x_i) \Delta x", color="#FFFF00")
        self.place_in_area(sum_sym, "A2", "A5", scale_factor=0.8)
        
        self.play(Create(rects_5), Write(sum_sym))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # "Summing infinite slivers reveals the exact whole."
        self.lecture[2].set_color("#FFFFFF")
        self.lecture[3].set_color("#FFFF00")
        
        # Transform the 5 thick rectangles into 50 thin yellow #FFFF00 rectangles
        rects_50 = axes.get_riemann_rectangles(
            curve, 
            x_range=[0, 4], 
            dx=0.08, 
            color="#FFFF00", 
            fill_opacity=0.4, 
            stroke_width=0.1,
            stroke_color="#FFFF00"
        )
        
        # Issue 30: Move int_sym to area A2-A5
        int_sym = MathTex(r"\int f(x) dx", color="#FFFF00")
        self.place_in_area(int_sym, "A2", "A5", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(rects_5, rects_50),
            ReplacementTransform(sum_sym, int_sym)
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # "It turns a changing rate back into a total amount."
        self.lecture[3].set_color("#FFFFFF")
        self.lecture[4].set_color("#00FF00")
        
        # Final visual anchor using Indicate (L004)
        self.play(
            Indicate(area, color="#00FF00"), 
            Indicate(int_sym, color="#00FF00"),
            Indicate(curve, color="#00FF00")
        )
        self.wait(2.0)
