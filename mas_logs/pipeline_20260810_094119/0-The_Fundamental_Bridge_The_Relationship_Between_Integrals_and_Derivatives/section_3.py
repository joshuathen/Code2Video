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
        self.setup_layout("Visual Proof: The Accumulation Function", [
            "Define area as a function of upper bound.",
            "Slide the upper bound to show accumulation.",
            "Derivative of area returns original function value.",
            "Drone tracks total biomass in a forest.",
            "Density at a point matches the derivative."
        ])
        
        # Setup Axes
        axes = Axes(
            x_range=[0, 5, 1], y_range=[0, 4, 1],
            axis_config={"include_tip": False}
        )
        
        graph_curve = axes.plot(lambda x: 0.2 * x**2 + 1, x_range=[0, 5], color=WHITE)
        area_group = VGroup(axes, graph_curve)
        
        # Apply layout fixes from requirements
        self.place_in_area(area_group, 'B3', 'E6', scale_factor=0.55)
        self.place_in_area(graph_curve, 'B3', 'E5', scale_factor=0.5)
        
        # Moving Line
        x_val = ValueTracker(0)
        moving_line = always_redraw(lambda: 
            axes.get_vertical_line(axes.c2p(x_val.get_value(), 0), color=YELLOW)
        )
        
        # Area under curve
        def get_area():
            return axes.get_area(graph_curve, x_range=[0, x_val.get_value()], color=BLUE, opacity=0.5)
            
        shaded_area = always_redraw(get_area)
        
        # Assets
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg").set_color("#87CEEB")
        forest = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/forest.svg").set_color("#87CEEB")
        
        area_label = Text("Area = F(x)", font_size=24, color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(area_group))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(shaded_area), Create(moving_line))
        self.play(x_val.animate.set_value(4), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(RED)
        self.place_at_grid(area_label, 'C4', scale_factor=0.7)
        self.play(Write(area_label))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        self.place_in_area(drone, 'E2', 'E2', scale_factor=0.5)
        self.place_in_area(forest, 'F2', 'F2', scale_factor=0.5)
        self.play(FadeIn(drone), FadeIn(forest))
        self.play(drone.animate.shift(RIGHT * 2))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PURPLE)
        density_point = Dot(color="#32CD32")
        self.place_at_grid(density_point, 'E3', scale_factor=1.2)
        self.play(GrowFromCenter(density_point))
        self.wait(2)
