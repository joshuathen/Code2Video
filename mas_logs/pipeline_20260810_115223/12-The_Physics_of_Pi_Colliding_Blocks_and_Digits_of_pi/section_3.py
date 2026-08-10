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
        self.setup_layout("The Counting Mechanism", [
            "Each collision marks a point on the arc.",
            "Collisions relate to the arc angle.",
            "Arc angle depends on mass ratios."
        ])
        
        # Counter setup
        counter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/counter.svg", color="#00FFFF")
        counter_val = ValueTracker(0)
        counter_text = Text("0", font_size=36, color="#00FFFF")
        
        counter_group = VGroup(counter_icon, counter_text).arrange(RIGHT)
        
        # Place counter
        self.place_at_grid(counter_group, "B5", scale_factor=0.8)
        
        # Velocity circle
        circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle, "C4", "F6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(counter_val.animate.set_value(1))
        # Update counter text logic
        dot = Dot(color="#00FFFF").move_to(circle.point_at_angle(PI/4))
        self.add(dot)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.play(counter_val.animate.set_value(5))
        arc = Arc(start_angle=0, angle=PI/2, radius=1.5, color="#00FFFF").move_to(circle.get_center())
        self.add(arc)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.play(counter_val.animate.set_value(10))
        counter_group.set_color("#FF00FF")
        self.play(counter_group.animate.scale(1.2).scale(1/1.2)) # Pulse effect
        self.wait(2)
