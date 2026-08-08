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
        self.setup_layout("Summary and Synthesis", [
            "Abstract spaces unify diverse data types.", 
            "Geometry provides intuition for complex objects.", 
            "Math DNA links functions and vectors."
        ])
        
        # Objects for animation
        concept1 = Text("Unification", font_size=24, color=WHITE)
        concept2 = Text("Geometric Intuition", font_size=24, color=WHITE)
        concept3 = Text("Shared DNA", font_size=24, color=WHITE)
        concept_group = VGroup(concept1, concept2, concept3)
        
        # Position objects as requested by Critic
        self.place_at_grid(concept1, 'B2', scale_factor=0.8)
        self.place_at_grid(concept2, 'C2', scale_factor=0.8)
        self.place_at_grid(concept3, 'D2', scale_factor=0.8)
        
        # Add a placeholder group for grid area as requested
        grid_group = VGroup(concept1, concept2, concept3)
        self.place_in_area(grid_group, 'B2', 'E5', scale_factor=0.9)
        
        # Ensure lecture text is placed appropriately
        self.place_in_area(self.lecture, 'A1', 'E1', scale_factor=0.75)
        
        self.add(concept1, concept2, concept3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), 
                  concept1.animate.set_color("#FFFF00"), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), 
                  concept2.animate.set_color("#FFFF00"), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"), 
                  concept3.animate.set_color("#FFFF00"), run_time=1)
        
        self.wait(2)
        self.play(FadeOut(self.lecture), FadeOut(concept1), FadeOut(concept2), FadeOut(concept3), FadeOut(self.title))
