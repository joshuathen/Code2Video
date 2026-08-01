import pathlib
from manim import *

# Pre-emptively create the directory to resolve the FileExistsError race condition 
# in the Text mobject's caching system (Manim CE v0.19.0).
pathlib.Path("media/texts").mkdir(parents=True, exist_ok=True)

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

class Section7Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Summary & The 'Golden Rule' Formula", 
            [
                'Use P to move from new to standard.', 
                'The inverse, P-inverse, takes us back again.', 
                'This formula links every perspective together.', 
                'Coordinates are relative, but the vector is absolute.', 
                'Choose the basis that makes your math easy.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        formula_std = Text(
            "[x]Std = P [x]New",
            color="#00FF00",
            font_size=32
        )
        # Resolved Issue 54: Move to row A and scale 1.0
        self.place_in_area(formula_std, "A1", "A6", scale_factor=1.0)
        
        self.play(Write(formula_std))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF8C00"))
        
        # Use Text instead of MathTex to avoid FileNotFoundError: 'latex'
        formula_new = Text(
            "[x]New = P^-1 [x]Std",
            color="#FF8C00",
            font_size=32
        )
        # Resolved Issue 55: Move to row B and scale 1.0
        self.place_in_area(formula_new, "B1", "B6", scale_factor=1.0)
        
        self.play(Write(formula_new))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Flowchart Boxes
        drone_box = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.6, width=2.2, color=WHITE),
            Text("Drone View", font_size=18, color=WHITE)
        )
        maya_box = VGroup(
            RoundedRectangle(corner_radius=0.1, height=0.6, width=2.2, color=WHITE),
            Text("Maya's View", font_size=18, color=WHITE)
        )
        
        self.place_at_grid(drone_box, "E2", scale_factor=1.0)
        self.place_at_grid(maya_box, "E5", scale_factor=1.0)
        
        # Arrow Drone -> Maya (P)
        arrow_top = Arrow(
            start=drone_box.get_right(), 
            end=maya_box.get_left(), 
            buff=0.1, 
            color=WHITE
        )
        # Use Text instead of MathTex
        label_p = Text("P", font_size=24, color=WHITE).next_to(arrow_top, UP, buff=0.1)
        
        self.play(FadeIn(drone_box), FadeIn(maya_box))
        self.play(GrowArrow(arrow_top), Write(label_p))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(WHITE))
        
        # Arrow Maya -> Drone (P^-1)
        arrow_bottom = CurvedArrow(
            start_point=maya_box.get_bottom() + LEFT*0.5,
            end_point=drone_box.get_bottom() + RIGHT*0.5,
            angle=-TAU/4,
            color=WHITE
        )
        # Use Text instead of MathTex
        label_pinv = Text("P^-1", font_size=24, color=WHITE).next_to(arrow_bottom, DOWN, buff=0.1)
        
        self.play(Create(arrow_bottom), Write(label_pinv))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # "Translator" Concept Label
        translator_text = Text("Basis = Mathematical Lens", font_size=24, color=YELLOW)
        # Resolved Issue 56: Use area F1-F6 and scale 0.8 to prevent clipping
        self.place_in_area(translator_text, "F1", "F6", scale_factor=0.8) 
        
        self.play(Write(translator_text))
        self.wait(3)
