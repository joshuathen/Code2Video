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
        self.setup_layout("Synthesis: The Transformer Core", [
            "Attention creates context-aware word embeddings.", 
            "This enables deep language understanding.", 
            "The foundation of modern Transformers."
        ])
        
        # Mobjects
        encoder_block = Rectangle(width=4, height=4, color=WHITE, fill_opacity=0.2)
        multi_head_block = Rectangle(width=2, height=1, color="#FFFF00", fill_opacity=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.place_in_area(encoder_block, 'B3', 'E5', scale_factor=0.7)
        self.play(FadeIn(encoder_block))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        self.place_at_grid(multi_head_block, 'C4', scale_factor=0.6)
        self.play(FadeIn(multi_head_block))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        # Animate data flow (simple particles)
        flow = VGroup(*[Dot(color=BLUE) for _ in range(5)]).arrange(RIGHT)
        self.place_at_grid(flow, 'F4', scale_factor=0.6)
        self.play(flow.animate.shift(UP * 2), run_time=2)
        self.lecture[2].set_color("#0000FF")
