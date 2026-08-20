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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Non-examples: The 'Breaking the Rules' Test", ["Some sets break rules.", "Missing the origin fails axioms.", "Therefore, it isn't a space."])
        
        # Non-linear set (a line not passing through origin)
        line = Line(start=[-1, 0, 0], end=[3, 4, 0], color="#FF0000")
        
        # Load Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color="#FF0000")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg", color="#FFFF00")
        
        label = Text("Non-linear Set", font_size=20, color="#FF0000")
        
        # Elements to add
        v1 = Dot(point=line.point_from_proportion(0.2), color="#FFFFFF")
        v2 = Dot(point=line.point_from_proportion(0.6), color="#FFFFFF")
        sum_v = Dot(point=np.array([1.5, 0.5, 0]), color="#FFFF00") # Forced off line
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.place_at_grid(compass, 'A1', scale_factor=0.3)
        self.play(Create(line), FadeIn(compass))
        self.place_at_grid(label, 'A6', scale_factor=0.6)
        self.play(FadeIn(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        self.place_at_grid(v1, 'B4', scale_factor=1.0)
        self.place_at_grid(v2, 'B4', scale_factor=1.0)
        self.play(FadeIn(v1), FadeIn(v2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(ruler, 'D4', scale_factor=0.4)
        arrow = Arrow(start=v1.get_center(), end=sum_v.get_center(), color="#FFFF00")
        self.play(GrowArrow(arrow), FadeIn(sum_v), FadeIn(ruler))
        self.wait(2)
