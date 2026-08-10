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
        self.setup_layout("The Derivative as a Sensitivity Operator", 
                          ["Think of derivatives as sensitivity operators.", 
                           "They transform functions into rate indicators.", 
                           "Like a robot's velocity command."])
        
        # Mobjects
        # B002: Group components
        axes = Axes(x_range=[-1, 5], y_range=[-1, 4], axis_config={"include_tip": False})
        func = axes.plot(lambda x: 0.25 * x**2, color=BLUE)
        x_val = ValueTracker(2.0)
        
        # Robot asset
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Persistent components
        # B010: Use updaters, not always_redraw for complex stuff
        def update_robot(mob):
            x0 = x_val.get_value()
            mob.move_to(axes.c2p(x0, 0.25 * x0**2))
            
        def update_tangent(mob):
            x0 = x_val.get_value()
            slope = 0.5 * x0
            y0 = 0.25 * x0**2
            mob.become(Line(start=axes.c2p(x0-0.5, y0-0.5*slope), end=axes.c2p(x0+0.5, y0+0.5*slope), color="#00FF00"))

        tangent = Line(color="#00FF00")
        tangent.add_updater(update_tangent)
        robot.add_updater(update_robot)
        
        group = VGroup(axes, func, tangent, robot)
        self.place_in_area(group, 'B3', 'E5', scale_factor=0.6)
        
        self.add(group)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(x_val.animate.set_value(3.0), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF6347"))
        self.play(x_val.animate.set_value(1.0), run_time=2)
        self.wait(1)
