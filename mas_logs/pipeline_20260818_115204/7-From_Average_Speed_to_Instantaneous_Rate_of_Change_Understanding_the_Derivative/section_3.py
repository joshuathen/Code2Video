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
        self.setup_layout("The Limit: Shrinking the Interval", ["The interval approaches zero.", "The secant line becomes a tangent.", "This slope is the instantaneous rate."])
        
        # Axes for the function
        axes = Axes(x_range=[0, 6], y_range=[0, 6], axis_config={"include_tip": False}).scale(0.5)
        self.place_in_area(axes, 'B2', 'E5')
        self.add(axes)
        
        # Parabola as f(x)
        curve = axes.plot(lambda x: 0.2 * x**2, color=BLUE)
        
        # Points and Secant
        x1, x2 = 1, 4
        p1 = Dot(axes.c2p(x1, 0.2*x1**2), color=YELLOW)
        p2 = Dot(axes.c2p(x2, 0.2*x2**2), color=YELLOW)
        
        secant = Line(p1.get_center(), p2.get_center(), color=YELLOW)
        
        self.add(curve, p1, p2, secant)

        # Asset note: The requested asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg is effectively empty/placeholder.
        # Adding a label for h -> 0 as requested in storyboard
        h_label = Text("h -> 0", color=WHITE, font_size=20)
        self.place_at_grid(h_label, 'A6')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Animate points merging
        self.play(
            p2.animate.move_to(axes.c2p(1.1, 0.2*1.1**2)),
            UpdateFromAlphaFunc(secant, lambda m, a: m.put_start_and_end_on(p1.get_center(), p2.get_center())),
            run_time=2
        )

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Change color to magenta and make tangent
        self.play(
            secant.animate.set_color("#FF00FF"),
            p2.animate.move_to(p1.get_center()),
            UpdateFromAlphaFunc(secant, lambda m, a: m.put_start_and_end_on(p1.get_center(), p1.get_center() + np.array([1, 0.4, 0]))),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        self.play(Flash(secant, color=GREEN, run_time=1))
        self.wait(1)
