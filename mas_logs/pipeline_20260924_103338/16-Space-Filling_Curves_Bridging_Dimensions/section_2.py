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
        self.setup_layout("The Peano Curve Construction", ["We begin with a simple line segment.", "Divide it into nine smaller sub-segments.", "Reconnect them to form a new pattern."])
        
        # === Animation for Lecture Line 1 ===
        # Draw a simple square iteration
        # VideoCritic Issue 24: reposition square to avoid overlap
        square = Square(side_length=2.4, color=WHITE)
        self.place_in_area(square, 'C2', 'E5', scale_factor=0.75)
        # Placeholder for [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # In this context, the square serves as the initial state
        self.play(Create(square))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Divide the square into nine equal sub-squares
        grid_lines = VGroup()
        for i in range(1, 3):
            # Vertical
            v_line = Line(start=square.get_left() + (i/3.0)*square.width*RIGHT + (square.height/2)*UP, 
                          end=square.get_left() + (i/3.0)*square.width*RIGHT + (square.height/2)*DOWN, color=WHITE)
            # Horizontal
            h_line = Line(start=square.get_bottom() + (i/3.0)*square.height*UP + (square.width/2)*LEFT, 
                          end=square.get_bottom() + (i/3.0)*square.height*UP + (square.width/2)*RIGHT, color=WHITE)
            grid_lines.add(v_line, h_line)
        
        self.play(Create(grid_lines))
        self.lecture[1].set_color("#FF00FF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Trace the Peano path through sub-squares
        # VideoCritic Issue 25: Reposition path
        path = VMobject(color="#00FFFF", stroke_width=4)
        path.set_points_smoothly([
            square.get_corner(DL) + 0.4*UR,
            square.get_corner(DL) + 0.4*RIGHT + 1.2*UP,
            square.get_corner(DL) + 1.2*RIGHT + 0.4*UP,
            square.get_corner(DL) + 1.2*RIGHT + 1.2*UP,
            square.get_corner(DL) + 1.2*RIGHT + 2.0*UP,
            square.get_corner(DL) + 2.0*RIGHT + 1.2*UP,
            square.get_corner(DL) + 2.0*RIGHT + 0.4*UP,
            square.get_corner(DL) + 2.0*RIGHT + 2.0*UP,
            square.get_corner(DL) + 0.4*RIGHT + 2.0*UP
        ])
        self.place_in_area(path, 'C3', 'E4', scale_factor=0.65)
        
        # VideoCritic Issue 26: Add explicit label
        label = Text("Peano Path", font_size=20, color="#00FFFF")
        self.place_at_grid(label, 'F3', scale_factor=0.5)

        self.play(Create(path), Write(label))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
