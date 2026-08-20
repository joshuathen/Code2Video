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
        lecture_lines = ["Conditional probability is probability given a condition.", "It shrinks the sample space to the condition.", "Example: knowing a card is red changes outcomes."]
        self.setup_layout("Prerequisite Warm-up: The Concept of Conditional Probability", lecture_lines)
        
        # Load Assets
        card_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/card.png")
        
        # Define elements
        sample_space = Rectangle(width=4, height=4, color="#FFFFFF")
        event_a = Circle(radius=1.2, color="#FF0000", fill_opacity=0.3)
        event_b = Circle(radius=1.2, color="#0000FF", fill_opacity=0.3)
        
        # Position them (Fixed per criticism)
        self.place_in_area(sample_space, 'A1', 'F3', scale_factor=0.6)
        self.place_at_grid(event_a, 'B2', scale_factor=0.7)
        self.place_at_grid(event_b, 'B4', scale_factor=0.7)
        
        intersection_rect = Intersection(event_a, event_b, color="#00FF00", fill_opacity=0.6)
        label = Text("P(A and B)", font_size=20, color="#00FF00")
        
        # Position fixed per criticism
        self.place_at_grid(label, 'D3', scale_factor=0.5)
        
        # Place asset
        self.place_at_grid(card_icon, 'F6', scale_factor=0.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(sample_space), FadeIn(card_icon), FadeIn(self.lecture[0]))
        self.lecture[0].set_color("#FFFF00")
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(event_a), FadeIn(event_b), FadeIn(self.lecture[1]))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(intersection_rect), FadeIn(label), FadeIn(self.lecture[2]))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
