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
        self.setup_layout("Synthesis: The 'Accumulation Function'", [
            "Define the accumulation function F(x).",
            "F(x) tracks area under curve.",
            "Rate of growth is f(x).",
            "Visualize the filling tank.",
            "Volume grows with flow rate."
        ])
        
        # Setup for math and visual objects
        axes = Axes(x_range=[0, 5], y_range=[0, 3], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2*x**2 + 0.5, color=BLUE)
        f_label = MathTex("f(x)", color=BLUE).next_to(curve, UP)
        
        graph_group = VGroup(axes, curve, f_label)
        # Apply fix for issue 28
        self.place_in_area(graph_group, 'A1', 'C4', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(Indicate(self.lecture[0], color=YELLOW))
        
        # === Animation for Lecture Line 2 ===
        self.play(Indicate(self.lecture[1], color=YELLOW))
        area = axes.get_area(curve, x_range=[0, 3], color="#32CD32", opacity=0.5)
        self.play(Create(area))
        
        # === Animation for Lecture Line 3 ===
        self.play(Indicate(self.lecture[2], color=YELLOW))
        deriv_text = MathTex("F'(x) = f(x)", color="#FF4500").scale(1.2)
        # Apply fix for issue 29
        self.place_at_grid(deriv_text, 'D5', scale_factor=0.9)
        self.play(Write(deriv_text))
        
        # === Animation for Lecture Line 4 ===
        self.play(Indicate(self.lecture[3], color=YELLOW))
        # Use asset from instruction: /scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg
        tank = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tank.svg")
        water = Rectangle(width=0.8, height=0.2, color=BLUE, fill_opacity=0.8).align_to(tank, DOWN)
        tank_group = VGroup(tank, water)
        # Apply fix for issue 30
        self.place_at_grid(tank_group, 'E3', scale_factor=0.7)
        self.play(FadeIn(tank), FadeIn(water))
        
        # === Animation for Lecture Line 5 ===
        self.play(Indicate(self.lecture[4], color=YELLOW))
        connection_arc = Arc(start_angle=PI/2, angle=PI/4, color=WHITE).add_tip()
        connection_arc.put_start_and_end_on(deriv_text.get_right(), tank.get_left())
        self.play(Create(connection_arc))
        self.play(water.animate.set_height(0.6), run_time=2)
