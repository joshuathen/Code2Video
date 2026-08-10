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
        self.setup_layout("Visualizing the Derivative: The Slope Tool", 
                          ["A derivative is a magnifying glass.", 
                           "It focuses on slope at one point.", 
                           "Watch the tangent line move across."])
        
        # Define the function and tangent
        axes = Axes(x_range=[-2, 2, 1], y_range=[-1, 5, 1], axis_config={"include_tip": False}).scale(0.5)
        func = axes.plot(lambda x: x**2 + 1, color=YELLOW)
        
        # Asset: magnifying glass icon
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg").scale(0.3)
        
        # Tangent line container (using ValueTracker)
        x_val = ValueTracker(-1.5)
        
        def get_tangent():
            x = x_val.get_value()
            y = x**2 + 1
            slope = 2 * x
            line = Line(start=axes.c2p(x - 0.5, slope * (x - 0.5 - x) + y),
                        end=axes.c2p(x + 0.5, slope * (x + 0.5 - x) + y),
                        color="#00BFFF")
            return line

        tangent = always_redraw(get_tangent)
        
        # Grouping for layout
        plot_group = VGroup(axes, func, tangent, magnifier)
        
        # Applying requested position adjustments
        self.place_in_area(plot_group, 'B3', 'E6', scale_factor=0.65)
        
        formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}", color=WHITE).scale(0.6)
        self.place_at_grid(formula, 'F5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00BFFF")
        # Position magnifier at start point
        magnifier.next_to(axes.c2p(-1.5, (-1.5)**2 + 1), UP+RIGHT, buff=0.1)
        self.play(Create(axes), Create(func), FadeIn(magnifier))
        self.play(Create(tangent))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF")
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00BFFF")
        # Animate tangent and move magnifier along curve
        self.play(
            x_val.animate.set_value(1.5), 
            UpdateFromFunc(magnifier, lambda m: m.move_to(axes.c2p(x_val.get_value(), x_val.get_value()**2 + 1) + UP*0.3+RIGHT*0.3)),
            run_time=3, 
            rate_func=linear
        )
        self.wait(1)
