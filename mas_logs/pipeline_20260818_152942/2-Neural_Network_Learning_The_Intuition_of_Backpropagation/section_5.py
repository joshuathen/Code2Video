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
        self.setup_layout("Putting It All Together: Gradient Descent", [
            "Gradient descent combines forward and backward steps.", 
            "The model makes a guess forward.", 
            "It adjusts weights backward via backpropagation.", 
            "This cycle repeats many times.", 
            "Loss minimizes as the robot learns."
        ])
        
        # Define objects
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 4, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: x**2, color=WHITE)
        
        # Applying layout improvements (Issue 32, 33, 39)
        self.place_in_area(axes, "C2", "F6", scale_factor=0.7)
        self.place_in_area(curve, "C2", "F6", scale_factor=0.6)
        
        # Use robot asset (Issue 19)
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        self.place_at_grid(robot, "D4", scale_factor=0.5) # Issue 34, 39
        
        # Loss counter
        loss_text = Text("Loss: High", font_size=20, color=WHITE).to_edge(RIGHT, buff=0.5).shift(UP*1)
        self.add(loss_text)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(axes), Create(curve), FadeIn(robot))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        # Showing adjustment down
        self.play(robot.animate.move_to(axes.c2p(-1.5, 2.25)))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.play(robot.animate.move_to(axes.c2p(-0.8, 0.64)))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        # Repetitive steps
        self.play(robot.animate.move_to(axes.c2p(-0.2, 0.04)), run_time=2)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(GREEN)
        # Minimize loss - highlight minimum (Issue 19)
        self.play(robot.animate.move_to(axes.c2p(0, 0)))
        loss_text.set_color(GREEN)
        loss_text.become(Text("Loss: Minimized", font_size=20, color=GREEN).move_to(loss_text.get_center()))
        self.wait(1)
