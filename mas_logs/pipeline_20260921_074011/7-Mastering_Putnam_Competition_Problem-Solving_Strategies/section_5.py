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
        self.setup_layout("Synthesis & Summary", [
            "Identify the structure of the problem.",
            "Choose your strategy: invariant or extremity.",
            "Structured exploration always beats brute force."
        ])
        
        # Animations
        # === Animation for Lecture Line 1 ===
        # Display 'Summary' title: #FFFFFF. Fades in.
        summary_text = Text("Summary", font_size=36, color=WHITE)
        self.place_at_grid(summary_text, "A3")
        self.play(FadeIn(summary_text))
        self.play(self.lecture[0].animate.set_color("#FF9900"))
        
        # List strategy keywords: #FF9900.
        kw1 = Text("Structure", font_size=28, color="#FF9900")
        self.place_in_area(kw1, 'C1', 'C2', scale_factor=0.6)
        self.play(Write(kw1))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF9900"))
        kw2 = Text("Strategy: Invariant/Extremity", font_size=28, color="#FF9900")
        self.place_in_area(kw2, 'C4', 'C6', scale_factor=0.6)
        self.play(Write(kw2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF9900"))
        kw3 = Text("Exploration > Brute Force", font_size=28, color="#FF9900")
        self.place_in_area(kw3, 'E1', 'E3', scale_factor=0.7)
        self.play(Write(kw3))
        
        # Final checkmark animation: #00FF00. Appears to confirm completion.
        checkmark = Checkmark(color="#00FF00")
        self.place_at_grid(checkmark, 'E5', scale_factor=0.5)
        self.play(Create(checkmark))
        self.wait(2)

class Checkmark(VMobject):
    def __init__(self, color=GREEN, **kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly([
            [-0.5, 0, 0],
            [-0.2, -0.5, 0],
            [0.5, 0.5, 0]
        ])
        self.set_stroke(color=color, width=8)
