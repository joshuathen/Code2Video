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
        self.setup_layout("Summary and Conclusion", [
            "Discrete collisions map to continuous geometry.",
            "Mathematics emerges from simple physical systems.",
            "Physics is the language of geometry."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Reiterate the blocks-to-pi connection graphically using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg].
        pi_symbol = Tex(r"$\pi$", font_size=96, color=YELLOW)
        equals = Tex(r"$\approx$", font_size=72, color=WHITE)
        collision_text = Tex(r"\text{Collisions}", font_size=48, color=BLUE)
        blocks_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg", color=WHITE)
        
        group = VGroup(blocks_icon, collision_text, equals, pi_symbol).arrange(RIGHT, buff=0.3)
        # Adjusted placement per instruction 37/44
        self.place_in_area(group, "A2", "D5", scale_factor=0.6)
        self.play(FadeIn(group))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Display the final ratio of collision counts.
        ratio_label = Text("1 : 1,000,000", font_size=36, color=GREEN)
        result_label = Text("= 3,141 Collisions", font_size=36, color=RED)
        ratio_group = VGroup(ratio_label, result_label).arrange(DOWN, buff=0.5)
        # Adjusted placement per instruction 38/44
        self.place_at_grid(ratio_group, "E4", scale_factor=0.7)
        self.play(FadeIn(ratio_group))
        self.lecture[1].set_color(GREEN)

        # === Animation for Lecture Line 3 ===
        # Fade out all elements to black.
        self.lecture[2].set_color(PURPLE)
        self.wait(2)
        self.play(FadeOut(group), FadeOut(ratio_group), FadeOut(self.lecture), FadeOut(self.title))
