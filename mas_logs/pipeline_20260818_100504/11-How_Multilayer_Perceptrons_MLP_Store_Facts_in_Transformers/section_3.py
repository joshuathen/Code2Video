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
        lecture_lines = [
            "MLP weights form a distributed system.",
            "Facts are stored as vector directions.",
            "Vectors merge into a resultant fact vector.",
            "This vector updates the token state.",
            "It helps locate facts within weights."
        ]
        self.setup_layout("Mathematical Mapping: The Superposition Concept", lecture_lines)
        
        # Load Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # Axes for vector visualization
        axes = Axes(x_range=[-1, 4], y_range=[-1, 4], axis_config={"include_tip": True}).scale(0.8)
        self.place_at_grid(axes, "D3", scale_factor=0.7)
        self.add(axes)
        
        self.place_at_grid(compass, "A6", scale_factor=0.5)
        self.add(compass)

        # Mobjects for animations
        vec1 = Vector([2, 1], color=BLUE)
        vec2 = Vector([1, 2], color=RED)
        resultant = Vector([3, 3], color=YELLOW)
        
        vec1_label = Text("Feature A", font_size=18, color=BLUE)
        vec2_label = Text("Feature B", font_size=18, color=RED)
        res_label = Text("Fact Vector", font_size=18, color=YELLOW)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        self.place_at_grid(vec1, "D3", scale_factor=0.7)
        self.place_at_grid(vec2, "D3", scale_factor=0.7)
        self.add(vec1, vec2)
        self.play(GrowArrow(vec1), GrowArrow(vec2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(protractor, "B5", scale_factor=0.5)
        self.add(protractor)
        self.place_in_area(resultant, "C2", "E4", scale_factor=0.6)
        self.play(
            GrowArrow(resultant),
            FadeIn(res_label.next_to(resultant.get_end(), UP))
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        dot = Dot(color=WHITE).move_to(axes.c2p(0,0))
        point_label = Text("Token State", font_size=16, color=WHITE)
        self.place_at_grid(point_label, "D5", scale_factor=0.5)
        self.add(dot, point_label)
        self.play(dot.animate.move_to(axes.c2p(3, 3)), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.play(Flash(resultant.get_end(), color=YELLOW))
        self.wait(2)
