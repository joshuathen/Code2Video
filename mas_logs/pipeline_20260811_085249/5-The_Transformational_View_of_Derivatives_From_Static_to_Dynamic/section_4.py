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
        self.setup_layout("Interpretation: The Derivative as a Sensitivity Mapper", 
                          ["Derivatives act as sensitivity mapping machines.", 
                           "Tiny input shifts trigger output changes.", 
                           "They quantify dynamic transformation factors.", 
                           "We link input growth to output.", 
                           "This reveals the system's sensitivity."])
        
        # Setup Axes
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], x_length=3, y_length=3)
        self.place_in_area(axes, 'B4', 'E6', scale_factor=0.7)
        self.add(axes)
        
        # Assets
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg", should_center=False)
        engine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/engine.svg", should_center=False)
        self.place_at_grid(microscope, 'B2', scale_factor=0.4)
        self.place_at_grid(engine, 'B3', scale_factor=0.4)
        
        # Function and derivative
        func = axes.plot(lambda x: 0.1 * x**3, color="#00FF00")
        deriv = axes.plot(lambda x: 0.3 * x**2, color="#FFFF00")
        
        # Tracking variables
        x_val = ValueTracker(0.5)
        
        # Use persistent mobjects for dynamic parts
        dot = Dot(color="#FFFFFF")
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_val.get_value(), 0.1 * x_val.get_value()**3)))
        
        tangent = Line(color="#FFFFFF", stroke_width=2)
        def update_tangent(l):
            x = x_val.get_value()
            y = 0.1 * x**3
            slope = 0.3 * x**2
            pt = axes.c2p(x, y)
            offset = 0.5
            l.put_start_and_end_on(
                axes.c2p(x - offset, y - offset * slope),
                axes.c2p(x + offset, y + offset * slope)
            )
        tangent.add_updater(update_tangent)
        
        self.add(func, deriv, dot, tangent, microscope, engine)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.play(FadeIn(func), FadeIn(microscope))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.play(x_val.animate.set_value(2.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.play(FadeIn(deriv), FadeIn(engine))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
