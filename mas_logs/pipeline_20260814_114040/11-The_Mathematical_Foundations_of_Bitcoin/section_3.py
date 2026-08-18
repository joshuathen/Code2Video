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
        self.setup_layout("Chaining the Blocks", ["Blocks chain together via previous hashes.", "This creates an immutable chronological history.", "Altering one link invalidates the entire chain."])
        
        block_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        
        # === Animation for Lecture Line 1 ===
        chaining_text = Text("Block Chaining", color=WHITE, font_size=30)
        self.place_at_grid(chaining_text, "A4", scale_factor=0.7)
        
        block1 = SVGMobject(block_asset).set_color("#FF4500")
        block2 = SVGMobject(block_asset).set_color("#FF6347")
        block3 = SVGMobject(block_asset).set_color("#FF7F50")
        
        self.place_at_grid(block1, "B2", scale_factor=0.6)
        self.place_at_grid(block2, "B3", scale_factor=0.6)
        self.place_at_grid(block3, "B4", scale_factor=0.6)
        
        self.play(FadeIn(chaining_text), FadeIn(block1), FadeIn(block2), FadeIn(block3))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        arrow1 = Arrow(start=block1.get_right(), end=block2.get_left(), color=WHITE, buff=0.1)
        arrow2 = Arrow(start=block2.get_right(), end=block3.get_left(), color=WHITE, buff=0.1)
        
        self.play(Create(arrow1), Create(arrow2))
        self.lecture[1].set_color(BLUE)

        # === Animation for Lecture Line 3 ===
        # Show invalidation
        self.play(block2.animate.set_color("#FF0000"))
        
        breaking_text = Text("INVALID", color=RED, font_size=20)
        breaking_text.next_to(block2, UP)
        
        self.play(Write(breaking_text))
        self.lecture[2].set_color(BLUE)
        self.wait(2)
