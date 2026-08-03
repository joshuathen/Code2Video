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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Algebra of Vectors: The 'Recipe'", 
                           ["We write vectors as a list of numbers.", 
                            "The numbers act as a recipe for movement.", 
                            "Component x moves right; component y moves up."])
        
        # === Animation for Lecture Line 1 ===
        # Display a #FFFFFF column matrix [4, 1] on the side, 
        # accompanied by [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/recipe.svg].
        matrix_41 = MathTex(r"\begin{bmatrix} 4 \\ 1 \end{bmatrix}", color="#FFFFFF")
        # Issue 25 Fix: Place at B3 instead of B1 for better visual balance
        self.place_at_grid(matrix_41, 'B3', scale_factor=1.2)
        
        # Issue 20 Fix: Load and place the recipe asset
        recipe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/recipe.svg")
        recipe_icon.set_color(WHITE)
        self.place_at_grid(recipe_icon, 'B2', scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(
            Write(matrix_41),
            FadeIn(recipe_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Trace a dashed #FFFFFF path 4 units right from origin.
        # Setup vector visual group components for positioning logic as per Issue 26
        dash_h = DashedLine(ORIGIN, RIGHT * 4, color="#FFFFFF")
        dash_v = DashedLine(RIGHT * 4, RIGHT * 4 + UP * 1, color="#FFFFFF")
        vector_arrow = Arrow(ORIGIN, RIGHT * 4 + UP * 1, color="#FF4500", buff=0)
        
        # Group elements to maintain internal spatial relationships
        vector_visual_group = VGroup(dash_h, dash_v, vector_arrow)
        # Issue 26 Fix: Use place_in_area for D3 to F6 at scale 1.0
        # We manually apply the transform to members to avoid using always_redraw
        self.place_in_area(vector_visual_group, 'D3', 'F6', scale_factor=1.0)
        
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        # Trace horizontal part of the "recipe"
        self.play(Create(dash_h))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Component x moves right; component y moves up. 
        # Draw the resulting #FF4500 vector arrow over the path.
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        # Trace vertical part and grow the resulting arrow
        self.play(Create(dash_v))
        self.play(GrowArrow(vector_arrow))
        self.wait(2)
