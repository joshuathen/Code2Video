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
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], x_length=4, y_length=4)
        self.place_in_area(axes, 'A1', 'E4', scale_factor=0.8)
        self.add(axes)
        
        # Function
        func = axes.plot(lambda x: 0.2 * x**3, color="#00FF00")
        self.add(func)
        
        # Dot and tangent
        x_val = ValueTracker(1)
        dot = Dot(color="#FFFFFF")
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_val.get_value(), 0.2 * x_val.get_value()**3)))
        self.add(dot)
        
        line = always_redraw(lambda: TangentLine(func, x_val.get_value(), length=1.5, color="#FFFFFF"))
        self.add(line)

        # Derivative curve
        deriv = axes.plot(lambda x: 0.6 * x**2, color="#FFFF00")
        self.add(deriv)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.play(x_val.animate.set_value(3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
