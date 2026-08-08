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
        self.setup_layout("Conclusion & Visualization", [
            "Mass growth reveals digits of Pi.",
            "Collision meter ticks: 3, 31, 314.",
            "Linking discrete events to continuous constants.",
            "The physics of Pi decoded.",
            "Simple rules, infinite complexity."
        ])
        
        # Assets
        blocks_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        weights_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weights.svg")
        
        # Elements
        collision_val = Integer(0)
        collision_label = Tex("Total Collisions:").scale(0.7)
        collision_group = VGroup(collision_label, collision_val).arrange(RIGHT)
        
        # Applying requested fixes
        self.place_at_grid(collision_group, 'C3', scale_factor=0.9)
        
        pi_display = MathTex(r"\pi \approx 3.14159...").scale(0.7)
        self.place_in_area(pi_display, 'D3', 'E5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(blocks_img, 'B5', scale_factor=0.5)
        self.play(FadeIn(blocks_img))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(weights_img, 'B2', scale_factor=0.5)
        self.play(FadeIn(weights_img))
        # Collision meter animation
        self.play(collision_val.animate.set_value(3), run_time=1)
        self.wait(0.5)
        self.play(collision_val.animate.set_value(31), run_time=1)
        self.wait(0.5)
        self.play(collision_val.animate.set_value(314), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(pi_display))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        self.play(Indicate(pi_display))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(FadeOut(collision_group), FadeOut(pi_display), FadeOut(blocks_img), FadeOut(weights_img))
        self.wait(2)
