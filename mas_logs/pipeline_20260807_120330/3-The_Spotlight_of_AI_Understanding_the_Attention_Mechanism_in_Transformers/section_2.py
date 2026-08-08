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
        # SECTION TITLE: Prerequisite: Words as Coordinates
        # LECTURE LINES:
        # 1. Machines understand words as vectors in multi-dimensional space.
        # 2. Similar words are placed physically closer together.
        # 3. This creates a mathematical map of word meanings.
        
        self.setup_layout(
            "Prerequisite: Words as Coordinates", 
            [
                "Machines understand words as vectors in multi-dimensional space.",
                "Similar words are placed physically closer together.",
                "This creates a mathematical map of word meanings."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Words 'King' and 'Queen' appear with vector coordinates. Color L1 to #FFFF00.
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        king = Text("King", font_size=24)
        king_vec = MathTex("[0.9, 0.1, ...]", font_size=18)
        queen = Text("Queen", font_size=24)
        queen_vec = MathTex("[0.8, 0.2, ...]", font_size=18)
        
        # Following Critic Fix for Issue 29: Use C2/C3 for Queen
        # This improves semantic consistency by starting similar words closer than the previous E5 position.
        self.place_at_grid(king, "B2")
        self.place_at_grid(king_vec, "B3") 
        self.place_at_grid(queen, "C2")
        self.place_at_grid(queen_vec, "C3") 
        
        self.play(
            FadeIn(king), FadeIn(king_vec),
            FadeIn(queen), FadeIn(queen_vec)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # 'King' and 'Queen' move near each other in space (#00FF00). Color L2 to #FFFF00.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        
        # Since they are already at B2 and C2 (relatively near), we'll shift them 
        # to confirm their 'proximity' and change their color to green as requested.
        self.play(
            king.animate.set_color("#00FF00"),
            king_vec.animate.set_color("#00FF00"),
            queen.animate.set_color("#00FF00"),
            queen_vec.animate.set_color("#00FF00"),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # The word 'Toaster' appears at a far corner (#FF0000). Color L3 to #FFFF00.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        toaster = Text("Toaster", font_size=24, color="#FF0000")
        toaster_vec = MathTex("[-0.5, 0.8, ...]", font_size=18, color="#FF0000")
        
        # Following Critic Fix for Issue 30: Use E5/E6 for Toaster to avoid Row F and cramped layout.
        self.place_at_grid(toaster, "E5")
        self.place_at_grid(toaster_vec, "E6")
        
        self.play(FadeIn(toaster), FadeIn(toaster_vec))
        self.wait(3)
