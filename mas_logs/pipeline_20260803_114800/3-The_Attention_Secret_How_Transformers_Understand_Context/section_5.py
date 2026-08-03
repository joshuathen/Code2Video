from manim import *
import numpy as np

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
        lecture_lines = [
            "Word embeddings evolve as they pass through Attention.",
            "They soak up the \"flavor\" of their neighboring words.",
            "'Bank' transforms from a building into a river edge.",
            "This memory of context enables coherent long-form writing.",
            "Attention reveals the secret of how AI understands language."
        ]
        self.setup_layout("The Outcome: Contextual Evolution", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Attention Layer - Gray Rectangle (Area B3 to E5)
        attention_layer = Rectangle(width=3.2, height=4.2, color="#444444", fill_opacity=0.3)
        self.place_in_area(attention_layer, "B3", "E5")
        
        # Issue 28: Move label to A4 to avoid overlap with layer boundary
        layer_label = Text("Attention Layer", font_size=20, color="#444444")
        self.place_at_grid(layer_label, "A4", scale_factor=0.8)
        
        # Word Box - White
        word_rect = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.7, color=WHITE, fill_opacity=0.1)
        word_text = Text("Bank", font_size=24, color=WHITE)
        word_box = VGroup(word_rect, word_text)
        
        # Issue 29: Changed starting position to D2 to avoid proximity to lecture text
        self.place_at_grid(word_box, "D2")
        
        self.play(FadeIn(attention_layer), FadeIn(layer_label))
        self.play(FadeIn(word_box))
        self.play(word_box.animate.move_to(self.grid["D4"]), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        
        # Neighbor words (Adjusted "River" to A5 to avoid layer_label at A4)
        neighbor_data = [("River", "A5"), ("Water", "F4"), ("Flow", "D6")]
        neighbors = VGroup()
        for text, pos in neighbor_data:
            n_rect = RoundedRectangle(corner_radius=0.1, width=1.1, height=0.5, color=BLUE, fill_opacity=0.1)
            n_text = Text(text, font_size=16, color=BLUE)
            n_grp = VGroup(n_rect, n_text)
            self.place_at_grid(n_grp, pos)
            neighbors.add(n_grp)
        
        self.play(FadeIn(neighbors))
        
        # Blue pulses from neighbors to central Word
        pulses = []
        for n in neighbors:
            p = Dot(n.get_center(), color=BLUE, radius=0.1)
            pulses.append(p)
            
        self.play(*(p.animate.move_to(word_box.get_center()).set_opacity(0) for p in pulses), run_time=1.5)
        self.play(word_rect.animate.set_fill(BLUE, opacity=0.4))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GOLD)
        
        # Transformation to Gold and Context Label
        # Issue 30: Context label moved to E4 to avoid overlap
        context_label = Text("Context: River Edge", font_size=18, color="#00FF00")
        self.place_at_grid(context_label, "E4", scale_factor=0.8)
        
        self.play(
            word_rect.animate.set_color(GOLD).set_fill(GOLD, opacity=0.5),
            word_text.animate.set_color(GOLD),
            FadeIn(context_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Word box moves out of the layer to signify output
        self.play(
            word_box.animate.move_to(self.grid["D6"]),
            context_label.animate.move_to(self.grid["E6"]),
            FadeOut(attention_layer),
            FadeOut(layer_label),
            FadeOut(neighbors),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Final highlight
        flash = Circle(radius=0.1, color=YELLOW, stroke_width=2).move_to(word_box.get_center())
        self.add(flash)
        self.play(flash.animate.scale(20).set_stroke(width=0).set_opacity(0), run_time=1)
        
        self.wait(2)
