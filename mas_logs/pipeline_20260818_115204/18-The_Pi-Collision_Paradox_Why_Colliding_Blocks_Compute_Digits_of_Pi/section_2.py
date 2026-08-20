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
        self.setup_layout("Prerequisite Physics: Conservation Laws", [
            "Conservation of momentum must hold true.",
            "Kinetic energy remains constant throughout collisions.",
            "These rules dictate block velocity changes."
        ])
        
        # --- Initialization of Elements ---
        # Momentum formula
        momentum_text = MathTex(r"p = mv", color="#FFD700", font_size=48)
        block_icon_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        momentum_group = VGroup(momentum_text, block_icon_1).arrange(RIGHT, buff=0.2)
        self.place_at_grid(momentum_group, 'B4', scale_factor=0.8)
        
        # Velocity arrow
        block_icon_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        vel_arrow = Arrow(start=ORIGIN, end=RIGHT*1.5, color=WHITE)
        vel_label = Text("v", font_size=24, color=WHITE).next_to(vel_arrow, UP, buff=0.1)
        velocity_group = VGroup(block_icon_2, vel_arrow, vel_label).arrange(RIGHT, buff=0.2)
        self.place_at_grid(velocity_group, 'C4', scale_factor=0.9)
        
        # Collision objects
        block1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        block2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg").set_color("#FF4500")
        collision_group = VGroup(block1, block2).arrange(RIGHT, buff=0.2)
        self.place_in_area(collision_group, 'E4', 'F6', scale_factor=0.8)

        # --- Animation Sequence ---

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(momentum_group))
        self.play(GrowArrow(vel_arrow), Write(vel_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF4500"))
        self.play(FadeIn(block1), FadeIn(block2))
        self.play(collision_group.animate.shift(RIGHT*0.2), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(
            momentum_text.animate.set_color(WHITE),
        )
        self.play(FadeOut(momentum_group), FadeOut(velocity_group), FadeOut(collision_group))
        self.wait(1)
