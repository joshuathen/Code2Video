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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Vectors live on a Cartesian grid.", "Arrows start at the origin point.", "Coordinates [x, y] define the endpoint."]
        self.setup_layout("Coordinate Representation (The Grid System)", lecture_lines)
        
        # Grid setup
        cartesian_grid = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3.2,
            y_length=3.2,
            axis_config={"color": "#D3D3D3"}
        )
        # Fix 38: Place grid in area
        self.place_in_area(cartesian_grid, 'B3', 'F6', scale_factor=0.9)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: The storyboard says to place a vector 'v' from origin to (2,3) in green (#00FF00)
        # Using SVGMobject for the referenced asset.
        asset_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(asset_icon, 'A1', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(cartesian_grid), FadeIn(asset_icon))
        self.lecture[0].set_color("#D3D3D3")

        # === Animation for Lecture Line 2 ===
        origin = cartesian_grid.c2p(0, 0)
        vec_end = cartesian_grid.c2p(2, 3)
        vector_arrow = Arrow(start=origin, end=vec_end, color="#00FF00", buff=0)
        # Fix 40: Scale/place arrow
        self.place_in_area(vector_arrow, 'C3', 'E5', scale_factor=1.0)
        
        self.play(Create(vector_arrow))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        dashed_x = DashedLine(cartesian_grid.c2p(2, 3), cartesian_grid.c2p(2, 0), color="#FFA500")
        dashed_y = DashedLine(cartesian_grid.c2p(2, 3), cartesian_grid.c2p(0, 3), color="#FFA500")
        
        coordinate_label = Text("[2, 3]", font_size=20, color="#FFA500")
        # Fix 39: Place coordinate label
        self.place_at_grid(coordinate_label, 'C4', scale_factor=0.7)
        
        self.play(Create(dashed_x), Create(dashed_y), Write(coordinate_label))
        self.lecture[2].set_color("#FFA500")
        
        self.wait(2)
