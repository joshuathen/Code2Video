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
        self.setup_layout("Dimension and Basis: The DNA of the Space", [
            "A basis is a set of fundamental building blocks.",
            "Every vector is a unique combination of basis elements.",
            "Basis vectors must be linearly independent from each other.",
            "The number of basis vectors defines the dimension.",
            "These coordinates allow us to navigate abstract spaces."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        red_circle = Circle(radius=0.3, color="#FF0000", fill_opacity=1)
        green_circle = Circle(radius=0.3, color="#00FF00", fill_opacity=1)
        blue_circle = Circle(radius=0.3, color="#0000FF", fill_opacity=1)
        
        self.place_at_grid(red_circle, "B2")
        self.place_at_grid(green_circle, "C2")
        self.place_at_grid(blue_circle, "D2")
        
        # Using Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/basis.svg
        basis_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/basis.svg", height=0.4, color=WHITE)
        basis_text = Text("Basis", font_size=24, color=WHITE)
        basis_header = VGroup(basis_icon, basis_text).arrange(RIGHT, buff=0.2)
        # Resolved Issue 29: Use place_in_area for basis header
        self.place_in_area(basis_header, "A2", "A3", scale_factor=0.8)
        
        self.play(
            FadeIn(red_circle), 
            FadeIn(green_circle), 
            FadeIn(blue_circle), 
            FadeIn(basis_header)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Trackers for RGB values (0.0 to 1.0)
        r_tracker = ValueTracker(0)
        g_tracker = ValueTracker(0)
        b_tracker = ValueTracker(0)
        
        # Sliders: line + dot
        r_line = Line(LEFT*0.4, RIGHT*0.4, color=GREY)
        self.place_at_grid(r_line, "B3")
        r_dot = Dot(color=WHITE).move_to(r_line.get_left())
        r_label = Text("R", font_size=16).next_to(r_line, LEFT, buff=0.1)

        g_line = Line(LEFT*0.4, RIGHT*0.4, color=GREY)
        self.place_at_grid(g_line, "C3")
        g_dot = Dot(color=WHITE).move_to(g_line.get_left())
        g_label = Text("G", font_size=16).next_to(g_line, LEFT, buff=0.1)

        b_line = Line(LEFT*0.4, RIGHT*0.4, color=GREY)
        self.place_at_grid(b_line, "D3")
        b_dot = Dot(color=WHITE).move_to(b_line.get_left())
        b_label = Text("B", font_size=16).next_to(b_line, LEFT, buff=0.1)
        
        # Result circle
        result_circle = Circle(radius=0.6, color=BLACK, fill_opacity=1)
        # Resolved Issue 30: Use place_in_area for result_circle
        self.place_in_area(result_circle, "B5", "D6", scale_factor=1.2)
        result_label = Text("Result", font_size=20, color=WHITE).next_to(result_circle, DOWN)
        
        # Updaters (Constraint 10/11: persistent mobjects, update in place)
        r_dot.add_updater(lambda d: d.move_to(r_line.point_from_proportion(r_tracker.get_value())))
        g_dot.add_updater(lambda d: d.move_to(g_line.point_from_proportion(g_tracker.get_value())))
        b_dot.add_updater(lambda d: d.move_to(b_line.point_from_proportion(b_tracker.get_value())))
        
        def update_result(c):
            r, g, b = r_tracker.get_value(), g_tracker.get_value(), b_tracker.get_value()
            new_color = rgb_to_color([r, g, b])
            c.set_fill(new_color)
            c.set_color(new_color)

        result_circle.add_updater(update_result)
        
        self.play(
            Create(r_line), FadeIn(r_dot), Write(r_label),
            Create(g_line), FadeIn(g_dot), Write(g_label),
            Create(b_line), FadeIn(b_dot), Write(b_label),
            FadeIn(result_circle), Write(result_label)
        )
        
        # Animate to Purple (#800080 is approx 0.5, 0, 0.5)
        self.play(
            r_tracker.animate.set_value(0.5),
            b_tracker.animate.set_value(0.5),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Highlight independence by scaling circles
        self.play(
            red_circle.animate.scale(1.2),
            green_circle.animate.scale(1.2),
            blue_circle.animate.scale(1.2),
        )
        self.play(
            red_circle.animate.scale(1/1.2),
            green_circle.animate.scale(1/1.2),
            blue_circle.animate.scale(1/1.2),
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        dim_text = Text("Dimension = 3", font_size=32, color="#D3D3D3")
        # Resolved Issue 31: Use place_in_area for dim_text
        self.place_in_area(dim_text, "E5", "F6", scale_factor=0.8)
        
        # Highlight R, G, B one by one with a rectangle
        rect = SurroundingRectangle(red_circle, color=WHITE, buff=0.1)
        self.play(Create(rect))
        self.wait(0.5)
        self.play(rect.animate.move_to(green_circle))
        self.wait(0.5)
        self.play(rect.animate.move_to(blue_circle))
        self.wait(0.5)
        self.play(FadeOut(rect), Write(dim_text))
        
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Navigate the space: move sliders to different values
        self.play(
            r_tracker.animate.set_value(0.9),
            g_tracker.animate.set_value(0.1),
            b_tracker.animate.set_value(0.9),
            run_time=1.5
        )
        self.play(
            r_tracker.animate.set_value(0.2),
            g_tracker.animate.set_value(0.8),
            b_tracker.animate.set_value(0.3),
            run_time=1.5
        )
        self.wait(2)
