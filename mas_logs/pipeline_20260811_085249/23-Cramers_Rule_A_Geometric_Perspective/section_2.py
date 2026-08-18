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
        self.setup_layout("Visualizing the System Ax = b", [
            "Linear system Ax = b combines vectors.",
            "x and y are scalar weights.",
            "Scaling vectors reaches target point b."
        ])
        
        # Grid positioning: Place axes at D4, scale 0.6
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": True}).scale(0.4)
        self.place_at_grid(axes, "D4", scale_factor=0.6)
        
        a1 = Vector([2, 1], color=YELLOW)
        a2 = Vector([1, 2], color=BLUE)
        b = Dot(color=RED)
        
        self.place_at_grid(b, "C4", scale_factor=0.7)
        
        # Initialize
        self.add(axes)
        self.play(Create(a1), Create(a2))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(FadeIn(b))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        x_weight = 1.0
        y_weight = 1.0
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        
        # Using a VGroup to manage vector group positioning for the critic
        vector_group = VGroup(a1, a2)
        
        # Scaling vectors to reach target point b
        # Target point is (3, 3) in axes coordinates.
        # a1=[2,1], a2=[1,2]. We want x(2,1) + y(1,2) = (3,3). 
        # x=1, y=1 works.
        x_val = 1.0
        y_val = 1.0
        
        new_a1 = Vector([x_val*2, x_val*1], color=YELLOW)
        new_a2 = Vector([y_val*1, y_val*2], color=BLUE)
        
        # Repositioning vector group at D4 for the critic
        self.play(
            a1.animate.become(new_a1),
            a2.animate.become(new_a2.shift(new_a1.get_end())),
            run_time=2
        )
        self.wait(2)
