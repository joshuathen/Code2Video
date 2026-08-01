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
        title = "The Challenge: Contact Tracing and Privacy"
        lecture_lines = [
            "- Contact tracing is crucial for controlling outbreaks.",
            "- Traditional methods raise privacy concerns.",
            "- Digital solutions can also compromise data.",
            "- A new approach is needed for privacy.",
            "- DP-3T offers a privacy-preserving alternative."
        ]
        self.setup_layout(title, lecture_lines)

        # Define colors for lecture lines and corresponding animations
        line_1_color = "#FFD700"  # Gold
        line_2_color = "#87CEEB"  # Sky Blue
        line_3_color = "#98FB98"  # Pale Green
        line_4_color = "#FF6347"  # Tomato
        line_5_color = "#BA55D3"  # Medium Orchid

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(line_1_color))
        contact_tracing_text = Text("Contact Tracing Challenges", font_size=36, color=line_1_color)
        self.place_at_grid(contact_tracing_text, 'B2', scale_factor=0.8) # Fix for Issue 24
        self.play(FadeIn(contact_tracing_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(line_2_color))
        privacy_concerns_text = Text("Privacy Concerns", font_size=36, color=line_2_color)
        self.place_at_grid(privacy_concerns_text, 'C2', scale_factor=0.8) # Fix for Issue 22
        self.play(FadeIn(privacy_concerns_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(line_3_color))
        need_new_solution_text = Text("Need for a New Solution", font_size=36, color=line_3_color)
        self.place_at_grid(need_new_solution_text, 'D2', scale_factor=0.8) # Fix for Issue 23
        self.play(FadeIn(need_new_solution_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # No animation specified in storyboard, only highlight the lecture line.
        self.play(self.lecture[3].animate.set_color(line_4_color))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # No animation specified in storyboard, only highlight the lecture line.
        self.play(self.lecture[4].animate.set_color(line_5_color))
        self.wait(1)

        self.wait(2) # Final pause
