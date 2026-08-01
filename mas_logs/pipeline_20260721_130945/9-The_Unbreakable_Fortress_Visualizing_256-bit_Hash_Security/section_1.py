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

class Section1Scene(TeachingScene):
    def construct(self):
        title_str = "Prerequisite: What is a Hash?"
        lecture_lines = [
            "Every piece of digital data has a unique fingerprint.",
            "We call this fingerprint a cryptographic hash.",
            "Even a tiny change creates a completely different output."
        ]
        self.setup_layout(title_str, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Color transition for lecture line
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # File icon - procedural
        file_rect = Rectangle(width=0.6, height=0.8, color=WHITE, fill_opacity=0.2)
        file_lines = VGroup(*[Line(LEFT*0.2, RIGHT*0.2, color=WHITE, stroke_width=2) for _ in range(3)])
        file_lines.arrange(DOWN, buff=0.1).move_to(file_rect.get_center())
        file_icon = VGroup(file_rect, file_lines)
        # Fix: Place file_icon at B2 instead of B3 for alignment
        self.place_at_grid(file_icon, "B2", scale_factor=0.8)

        # Hash text
        hash_text_1 = Text("a1b2c3d4", font="Monospace", color="#FFFF00")
        # Fix: Place hash_text_1 at B4 instead of B5 for alignment
        self.place_at_grid(hash_text_1, "B4", scale_factor=0.7)

        arrow_1 = Arrow(
            start=file_icon.get_right(),
            end=hash_text_1.get_left(),
            buff=0.2,
            color=WHITE
        )

        self.play(FadeIn(file_icon))
        self.play(Create(arrow_1))
        self.play(Write(hash_text_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Clear previous for clarity but keep the concept
        self.play(FadeOut(file_icon, arrow_1, hash_text_1))

        # Movie block and Word block
        movie_block = VGroup(
            Rectangle(width=1.2, height=0.5, color=BLUE, fill_opacity=0.3),
            Text("MOVIE.MP4", font_size=18, color=WHITE)
        )
        word_block = VGroup(
            Rectangle(width=1.2, height=0.5, color=GREEN, fill_opacity=0.3),
            Text("HELLO", font_size=18, color=WHITE)
        )

        # Fix: Adjust rows from D/E to C/D to improve vertical spacing
        self.place_at_grid(movie_block, "C2", scale_factor=0.8)
        self.place_at_grid(word_block, "D2", scale_factor=0.8)

        hash_out_movie = Text("f7e6d5c4", font="Monospace", color="#00FFFF")
        hash_out_word = Text("b1a29384", font="Monospace", color="#00FFFF")

        # Fix: Adjust rows from D/E to C/D
        self.place_at_grid(hash_out_movie, "C4", scale_factor=0.7)
        self.place_at_grid(hash_out_word, "D4", scale_factor=0.7)

        arrow_movie = Arrow(movie_block.get_right(), hash_out_movie.get_left(), color=WHITE)
        arrow_word = Arrow(word_block.get_right(), hash_out_word.get_left(), color=WHITE)

        self.play(
            FadeIn(movie_block),
            FadeIn(word_block)
        )
        self.play(
            Create(arrow_movie),
            Create(arrow_word)
        )
        self.play(
            Write(hash_out_movie),
            Write(hash_out_word)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF8800"))

        # Modify word block
        word_block_mod = VGroup(
            Rectangle(width=1.2, height=0.5, color=GREEN, fill_opacity=0.3),
            Text("HELL0", font_size=18, color=WHITE) # Changed O to 0
        )
        # Fix: Adjust row to D2
        self.place_at_grid(word_block_mod, "D2", scale_factor=0.8)

        hash_out_word_mod = Text("39d182f0", font="Monospace", color="#FF8800")
        # Fix: Adjust row to D4
        self.place_at_grid(hash_out_word_mod, "D4", scale_factor=0.7)

        # Indication of change
        self.play(Indicate(word_block[1]))
        self.play(Transform(word_block, word_block_mod))
        
        # Hash changes drastically
        self.play(
            hash_out_word.animate.set_color("#FF8800"),
            Transform(hash_out_word, hash_out_word_mod)
        )
        self.play(Indicate(hash_out_word))
        
        self.wait(2)
