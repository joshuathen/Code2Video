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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Blocks link via the previous block's hash.", "Altering one block invalidates the entire chain.", "This ensures security through mathematical immutability."]
        self.setup_layout("The Immutable Chain: Blockchain Integration", lecture_lines)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg
        block_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        
        blocks = VGroup()
        for i in range(3):
            # We don't scale here, let place_at_grid handle scaling.
            block = SVGMobject(block_path, color=WHITE)
            block.set_stroke(WHITE, width=2)
            blocks.add(block)
        
        # Updated per VideoCritic feedback (Issues 30, 31, 32)
        self.place_at_grid(blocks[0], 'D1', scale_factor=0.7)
        self.place_at_grid(blocks[1], 'D3', scale_factor=0.7)
        self.place_at_grid(blocks[2], 'D5', scale_factor=0.7)
        
        arrows = VGroup()
        for i in range(2):
            arrow = Arrow(start=blocks[i].get_right(), end=blocks[i+1].get_left(), buff=0.1, color=WHITE, stroke_width=4)
            arrows.add(arrow)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(blocks[0]), FadeIn(blocks[1]), FadeIn(arrows[0]))
        self.play(FadeIn(blocks[2]), FadeIn(arrows[1]))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        
        # Highlight: Altering blocks turns everything RED
        self.play(
            blocks.animate.set_color(RED),
            arrows.animate.set_color(RED)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        
        # Highlighting the hash link in green
        self.play(arrows.animate.set_color(GREEN))
        
        self.wait(2)
