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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize the layout with title and lecture lines
        # Mandatory call to setup_layout
        self.setup_layout(
            "Summary & Key Takeaway", 
            [
                "Remember, y is always a hidden function of x.", 
                "Differentiate, apply Chain Rule, then isolate your derivative.", 
                "Now you can find slopes for any tangled equation."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Visual: Show the identity 'd/dx [y] = dy/dx' in a gold frame (#FFD700).
        self.lecture[0].set_color("#FFD700")
        identity_tex = Text("d/dx[y] = dy/dx", color="#FFD700")
        frame = SurroundingRectangle(identity_tex, color="#FFD700", buff=0.2)
        identity_group = VGroup(identity_tex, frame)
        
        # Position in the top section of the right-side grid (Rows A and B)
        # Resolved Issue 46: scale_factor=1.1
        self.place_in_area(identity_group, "A2", "B5", scale_factor=1.1)
        self.play(Create(frame), Write(identity_tex))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: Display a checklist with 'Diff Both Sides', 'Chain Rule', and 'Isolate dy/dx' in green (#00FF00).
        self.lecture[1].set_color("#00FF00")
        check1 = Text("✓ Diff Both Sides", color="#00FF00", font_size=24)
        check2 = Text("✓ Chain Rule", color="#00FF00", font_size=24)
        check3 = Text("✓ Isolate dy/dx", color="#00FF00", font_size=24)
        checklist = VGroup(check1, check2, check3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Position in the middle section of the grid (Rows C to E)
        # Resolved Issue 47: Area 'C2' to 'E5', scale_factor=1.1
        self.place_in_area(checklist, "C2", "E5", scale_factor=1.1)
        self.play(Write(checklist))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visual: Fade in the closing text 'Implicit slopes revealed!' in white (#FFFFFF).
        self.lecture[2].set_color("#FFFFFF")
        closing_text = Text("Implicit slopes revealed!", color="#FFFFFF", font_size=32)
        
        # Position in the bottom section of the grid (Row F)
        # Resolved Issue 48: Area 'F2' to 'F5'
        self.place_in_area(closing_text, "F2", "F5", scale_factor=1.0)
        self.play(FadeIn(closing_text))
        self.wait(3)
