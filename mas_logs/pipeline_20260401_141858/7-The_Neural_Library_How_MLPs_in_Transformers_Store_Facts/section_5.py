from manim import *
import numpy as np

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
        # Data for setup
        title = "Information Injection into the Residual Stream"
        lines = [
            "Information travels along a path called the residual stream.",
            "The MLP does not replace the existing data flow.",
            "Instead, it adds new factual vectors to the stream.",
            "This process is represented by simple vector addition.",
            "Each layer enriches the context with more specific facts."
        ]
        self.setup_layout(title, lines)

        # Colors
        STREAM_COLOR = "#ADD8E6"
        DATA_COLOR = BLUE_A
        MLP_COLOR = GOLD_A
        ADD_COLOR = GREEN_A
        FACT_COLOR = YELLOW_A

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(STREAM_COLOR)
        
        belt = Rectangle(width=5.5, height=0.4, color=STREAM_COLOR, fill_opacity=0.3)
        self.place_in_area(belt, "C1", "C6")
        
        belt_label = Text("Residual Stream", font_size=20, color=STREAM_COLOR)
        # Fixed Issue 55: Use place_in_area to center over the belt
        self.place_in_area(belt_label, "B1", "B6", scale_factor=0.8)
        
        # Stream flow dots
        dots = VGroup(*[Dot(radius=0.06, color=STREAM_COLOR) for _ in range(6)])
        for i, dot in enumerate(dots):
            dot.move_to(belt.get_left() + RIGHT * i * 1.1)
            
        def flow_update(mob, dt):
            for d in mob:
                d.shift(RIGHT * dt * 0.7)
                if d.get_x() > belt.get_right()[0]:
                    d.set_x(belt.get_left()[0])

        self.play(Create(belt), Write(belt_label))
        self.add(dots)
        dots.add_updater(flow_update)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(DATA_COLOR)
        
        # Use Text instead of MathTex to avoid LaTeX dependency errors
        data_x = Text("x", color=DATA_COLOR, font_size=36, slant=ITALIC)
        self.place_at_grid(data_x, "C1")
        
        self.play(FadeIn(data_x))
        self.play(data_x.animate.move_to(self.grid["C3"]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(MLP_COLOR)
        
        mlp_box = RoundedRectangle(corner_radius=0.1, width=2.0, height=1.0, color=MLP_COLOR, fill_opacity=0.2)
        self.place_in_area(mlp_box, "E2", "E3")
        mlp_text = Text("MLP", font_size=24, color=MLP_COLOR)
        self.place_in_area(mlp_text, "E2", "E3")
        
        tap_arrow = Arrow(self.grid["C2"], self.grid["E2"], color=MLP_COLOR, buff=0.1)
        
        self.play(Create(mlp_box), Write(mlp_text))
        self.play(Create(tap_arrow))
        
        # Using Unicode Delta for Delta x
        delta_x = Text("Δx", color=MLP_COLOR, font_size=36, slant=ITALIC)
        self.place_at_grid(delta_x, "E4")
        self.play(Write(delta_x))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ADD_COLOR)
        
        add_arrow = Arrow(self.grid["E4"], self.grid["C4"], color=ADD_COLOR, buff=0.1)
        plus_icon = Text("+", font_size=32, color=ADD_COLOR)
        self.place_at_grid(plus_icon, "C4")
        
        # Using Text for formula fallback
        formula = Text("x_new = x + Δx", font_size=32, color=ADD_COLOR, slant=ITALIC)
        # Fixed Issue 53: Use place_in_area for long text
        self.place_in_area(formula, "B3", "B6", scale_factor=0.6)
        
        self.play(Create(add_arrow))
        self.play(Write(plus_icon))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(FACT_COLOR)
        
        combined_data = Text("x + Δx", color=FACT_COLOR, font_size=36, slant=ITALIC)
        # Fixed Issue 54: Use place_in_area to avoid direct overlap and handle width
        self.place_in_area(combined_data, "C4", "C5", scale_factor=0.8)
        
        self.play(
            FadeOut(data_x),
            FadeOut(plus_icon),
            FadeIn(combined_data)
        )
        
        self.play(combined_data.animate.move_to(self.grid["C6"]), run_time=2)
        
        # Cleanup
        dots.remove_updater(flow_update)
        self.wait(2)
