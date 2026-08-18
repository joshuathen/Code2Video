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
        self.setup_layout("The Concept of Linear Dependence", [
            "Linear dependence means vectors are collinear.",
            "Redundant vectors add no new span reach.",
            "Example: two drones flying in same direction."
        ])
        
        # Assets
        drone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/drone.svg", color=WHITE)
        
        v = Vector([1, 1], color=BLUE)
        w = Vector([2, 2], color=YELLOW)
        
        v_label = MathTex("\\vec{v}", color=BLUE)
        w_label = MathTex("\\vec{w} = 2\\vec{v}", color=YELLOW)
        
        self.place_at_grid(v, "B2", scale_factor=0.8)
        self.place_at_grid(w, "B5", scale_factor=0.8)
        
        v_label.next_to(v.get_end(), UP)
        w_label.next_to(w.get_end(), DOWN)
        
        drone_group = drone.copy().scale(0.3)
        self.place_at_grid(drone_group, "A3")
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(v), Create(w), Write(v_label), Write(w_label), FadeIn(drone_group))
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        span_area = Rectangle(width=3, height=2, color=WHITE, fill_opacity=0.2)
        self.place_in_area(span_area, "D2", "F5", scale_factor=0.9)
        self.play(FadeIn(span_area), self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeOut(w), FadeOut(w_label), self.lecture[2].animate.set_color(ORANGE))
        self.wait(2)
